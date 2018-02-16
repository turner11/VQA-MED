using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Drawing.Imaging;
using VQA.Logic;
using System.IO;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using System.Windows.Navigation;
using System.Threading.Tasks;
using VQA.GUI;

namespace SDKSamples.ImageSample
{
    public sealed partial class MainWindow : Window
    {

        public PhotoCollection Photos;
        public static List<(string Images, string Captions, string PixelMaps, string pythonHandler)> KnownDataLocations
        {
            get
            {
                return new List<(string, string, string,string )>
                                {(@"C:\Users\Public\Documents\Data\2017\val2017", 
                                 @"C:\Users\Public\Documents\Data\2017\annotations\stuff_val2017.json",
                                  @"C:\Users\Public\Documents\Data\2017\annotations\stuff_val2017_pixelmaps",
                                  @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py"),

                                    (@"C:\Users\Public\Documents\Data\2014 Train\train2014", 
                                    @"C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json",
                                    "",
                                    @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA14.py"),
                                 // (Environment.CurrentDirectory + "\\images","","")
                                    };
            }
        }

        internal UIElementGenerator UiElementGenerator { get; }

        private VqaLogics logics;

        public MainWindow()
        {
            InitializeComponent();
            this.UiElementGenerator = new UIElementGenerator(this);

            this.SetNextFolder();
            
        }

        private void OnPhotoClick(object sender, RoutedEventArgs e)
        {
            PhotoView pvWindow = new PhotoView();
            var photo = (Photo)lstPhotos.SelectedItem;
            pvWindow.SelectedPhoto = photo;
            pvWindow.Show();
        }

        private void editPhoto(object sender, RoutedEventArgs e)
        {
            PhotoView pvWindow = new PhotoView();
            pvWindow.SelectedPhoto = (Photo)lstPhotos.SelectedItem;
            pvWindow.Show();
        }

        private void OnImagesDirChangeClick(object sender, RoutedEventArgs e)
        {
            this.setCurrnetDataPaths();
        }

        private void setCurrnetDataPaths()
        {
            (string Images, string Captions, string PixelMaps,string pythonHandler) = KnownDataLocations.Where(tpl => String.Equals(tpl.Images, ImagesDir.Text, StringComparison.InvariantCulture)).FirstOrDefault();
            this.logics = new VqaLogics(Captions, PixelMaps, pythonHandler);
            this.Photos.Path = Images;
            this.Photos.Captions = Captions;
            this.Photos.PixelMaps = PixelMaps;
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            //SetNextFolder();

            this.setCurrnetDataPaths();
        }

        internal void SetNextFolder()
        {
            (string Images, string Captions, string PixelMap, string pythonHandler) = this.GetNextFolder();
            ImagesDir.Text = Images;
        }

        private (string Images, string Captions, string PixelMap, string pythonHandler) GetNextFolder()
        {
            var allItems = KnownDataLocations;
            var currImagesFolder = this.ImagesDir.Text;
            var currIdx = allItems.FindIndex(tpl => tpl.Images == currImagesFolder);
            var nextIdx = currIdx + 1;
            var nextItem = allItems[nextIdx % allItems.Count];
            return nextItem;
        }

        private void ImagesDir_MouseDoubleClick(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            SetNextFolder();

        }

        private async void btnAsk_Click(object sender, RoutedEventArgs e)
        {
         

            bool query = true;
            if (query)
            {
                await this.QueryImaeghs();
            }
            else
                await this.Ask();
           

        }

        private async Task Ask()
        {
            var question = this.txbQuestion.Text;
            var imagePath = (this.lstPhotos.SelectedItem as Photo)?.Path ?? "";
            var imagesDirectory = this.ImagesDir.Text;
            var hasData = !String.IsNullOrWhiteSpace(question) && File.Exists(imagePath) && Directory.Exists(imagesDirectory);
            string responce;
            if (!hasData)            
                responce = "Got invalid data to query";
            
            else            
                responce = await this.logics.Ask(question, new FileInfo(imagePath));
            

            Debug.WriteLine(responce);
            this.txbResponce.Text = responce;
        }

        private async Task QueryImaeghs()
        {
            var question = this.txbQuestion.Text;
            if (question.Length == 0)
            {
                this.Photos.Filter = null;
                return;
            }

            var match_images = await this.logics.Query(question);            
            this.Photos.Filter = fn => match_images.Contains(fn);
            
        }

        private void txbQuestion_KeyDown(object sender, System.Windows.Input.KeyEventArgs e)
        {
            var isControlPressed = e.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control;
            var isEnter = e.Key == System.Windows.Input.Key.Enter;
            if (isControlPressed && isEnter)
                this.btnAsk.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        }

        private async void lstPhotos_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            this.spImageData.Children.Clear();
            var photo = lstPhotos.SelectedItem as Photo;
            if (photo == null || this.logics == null)
                return;

            var caption = "";
            try
            {
                
                var dataDict = await this.logics.GetImageData(photo.Path);
                this.spImageData.Children.Clear(); //Clearing again because maybe we got a new request in the meanwhile
                var allitems = dataDict.OrderByDescending(pair => pair.Key.ToLower() == "caption" || pair.Key.ToLower() == "pixel map").ToList();
                foreach (var pair in allitems)
                {
                    (var headerItem, var contentItem) = this.UiElementGenerator.GetDataItemsControls(pair.Key, pair.Value);
                    this.spImageData.Children.Add(headerItem);
                    this.spImageData.Children.Add(contentItem);
                }
                //caption = await VqaLogics.GetImageCaptions(this.Photos.Captions, photo.Path);
            }
            catch (Exception ex)
            {
                caption = ex.Message;
            }
            
            this.txbResponce.Text = caption;
        }        

       

        private void Window_Activated(object sender, EventArgs e)
        {
        }
    }
}
