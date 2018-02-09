using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using SDKSamples.ImageSample;
using VQA.Logic;

namespace VQA.GUI
{
    internal class UIElementGenerator
    {
        private readonly MainWindow _parent;

        public UIElementGenerator(SDKSamples.ImageSample.MainWindow parent)
        {
            this._parent = parent;
        }

        public (UIElement header, UIElement contentItem) GetDataItemsControls(string header, object child)
        {
            Label headerItem = this.GetParentControl(header);      
           
            var contentItem = this.GetChildControl(child);
            
            return (header: headerItem, contentItem: contentItem);



        }

        private Label GetParentControl(string headerValue)
        {
            var headerItem = new Label() { Style = this._parent.Resources["MetadataHeader"] as Style, Content = headerValue };
            if (headerValue.ToLower() == VqaLogics.ERROR_KEY)
                headerItem.Background = new SolidColorBrush(Colors.Red);
            return headerItem;
        }

        private UIElement GetChildControl(System.Collections.IEnumerable items)
        {
            var lv = new ListView();
            
            
            foreach (var item in items)
            {
                var currElement = this.GetChildControl(item);
                lv.Items.Add(currElement);
                //var lbi = new ListBoxItem();
                //lbi.Content = currElement;
                //lv.Items.Add(lbi);

            }
            return lv;
        }

        private UIElement GetChildControl(object item)
        {
            UIElement el;
            if (item is System.Collections.IEnumerable en && !(item is IEnumerable<char>) && !(item is Newtonsoft.Json.Linq.JValue))
                el = this.GetChildControl(en);            
            else
                el = this.GetChildControl(item.ToString());

            return el;
        }

        private UIElement GetChildControl(string item)
        {
            
            if (File.Exists(item) && new string[] { "jpg","png"}.Any(ext=> item.ToLower().EndsWith(ext)))
                return this.GetImageChildControl(item);
            else
                return this.GetTextChildControl(item);



        }

        private UIElement GetTextChildControl(string item)
        {

            var contentItem = new TextBlock() { Text = item, TextWrapping = TextWrapping.Wrap };

            var cMenu = new ContextMenu();
            if (item.ToLower().StartsWith("http"))
            {
                var miOpenUrl = new MenuItem();
                cMenu.Items.Add(miOpenUrl);

                //contentItem.Text = "";
                var hyperlink = new Hyperlink() { NavigateUri = new Uri(item) };
                hyperlink.RequestNavigate += new System.Windows.Navigation.RequestNavigateEventHandler(this.hyperlink_RequestNavigate); //to be implemented
                contentItem.Inlines.Add(hyperlink);

                //contentItem.MouseDown += this.ContentItem_MouseDown;
                miOpenUrl.Header = "Open URL";
                miOpenUrl.Click += (sender, args) =>
                {
                    var url = item;
                    try
                    {
                        Process.Start(url);
                    }
                    catch (System.ComponentModel.Win32Exception)
                    {
                        Process.Start("chrome.exe", url);
                    }
                };


            }



            var miTextToClipBoard = new MenuItem();
            cMenu.Items.Add(miTextToClipBoard);

            miTextToClipBoard.Header = "Copy text to clipboard";
            miTextToClipBoard.Click += (sender, args) => Clipboard.SetText(item);


            contentItem.ContextMenu = cMenu;
            return contentItem;
        }

        private UIElement GetImageChildControl(string path_to_image)
        {
            var img = new Image();
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.UriSource = new Uri(path_to_image, UriKind.RelativeOrAbsolute);
            bmp.EndInit();

            img.Source = bmp;
            //img.Width = bmp.PixelWidth;

            return img;
        }

        private void ContentItem_MouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            var txb = sender as TextBlock;
            if (txb == null)
                return;
            var url = txb.Text;

            try
            {
                Process.Start(url);
            }
            catch (System.ComponentModel.Win32Exception)
            {
                Process.Start("chrome.exe", url);
            }

        }

        private void hyperlink_RequestNavigate(object sender, RequestNavigateEventArgs e)
        {
            throw new NotImplementedException();
        }
    }
}
