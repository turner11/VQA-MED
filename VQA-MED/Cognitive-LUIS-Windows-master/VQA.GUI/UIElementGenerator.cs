using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
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


            var contentItem = this.GetChildControl(child.ToString());
            
            return (header: headerItem, contentItem: contentItem);



        }

        private Label GetParentControl(string headerValue)
        {
            var headerItem = new Label() { Style = this._parent.Resources["MetadataHeader"] as Style, Content = headerValue };
            if (headerValue.ToLower() == VqaLogics.ERROR_KEY)
                headerItem.Background = new SolidColorBrush(Colors.Red);
            return headerItem;
        }

        private UIElement GetChildControl(string item)
        {
            var contentItem = new TextBlock() { Text = item, TextWrapping = TextWrapping.Wrap };

            if (item.ToLower().StartsWith("http"))
            {
                //contentItem.Text = "";
                var hyperlink = new Hyperlink() { NavigateUri = new Uri(item) };
                hyperlink.RequestNavigate += new System.Windows.Navigation.RequestNavigateEventHandler(this.hyperlink_RequestNavigate); //to be implemented
                contentItem.Inlines.Add(hyperlink);
                contentItem.MouseDown += this.ContentItem_MouseDown;
            }

            return contentItem;
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
