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
        public static List<VqaData> KnownDataLocations
        {
            get
            {
             
                var vqa2017 = new VqaData(images: @"C:\Users\Public\Documents\Data\2017\val2017"
                                 , captions: @"C:\Users\Public\Documents\Data\2017\annotations\stuff_val2017.json"
                                 , pixelMaps: @"C:\Users\Public\Documents\Data\2017\annotations\stuff_val2017_pixelmaps"
                                 , pythonHandler: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py");

                var vqa2014 = new VqaData(images: @"C:\Users\Public\Documents\Data\2014 Train\train2014"
                                    , captions: @"C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json"
                                    , pixelMaps: ""
                                    , pythonHandler: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA14.py");

                var vqa2015 = new VqaData(images: vqa2014.Images
                                         , captions: "D:\\GitHub\\VQA-Keras-Visual-Question-Answering\\data\\Questions_Train_mscoco\\MultipleChoice_mscoco_train2014_questions.json"
                                         , pixelMaps: ""
                                         , pythonHandler: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA14_multiple.py");
                return new List<VqaData>{ vqa2015,vqa2017, vqa2014};
            }
        }

        internal UIElementGenerator UiElementGenerator { get; }

        private VqaLogics logics;

        public MainWindow()
        {
            InitializeComponent();
            this.UiElementGenerator = new UIElementGenerator(this);

            
        }

        private void Init_CmbImages()
        {
            var data_locations = KnownDataLocations;
            this.ImagesDir.ItemsSource = data_locations;
            var dummy = new VqaData("", "", "", "");
            this.ImagesDir.DisplayMemberPath = nameof(dummy.Images);
            this.ImagesDir.SelectedIndex = 0;
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
            //this.setCurrnetDataPaths();
        }

        private void setCurrnetDataPaths()
        {
            if (this.ImagesDir.SelectedItem is VqaData vqaData)
            {
                this.logics = new VqaLogics(vqaData.Captions, vqaData.PixelMaps, vqaData.PythonHandler);
                this.Photos.Path = vqaData.Images;
                this.Photos.Captions = vqaData.Captions;
                this.Photos.PixelMaps = vqaData.PixelMaps;
            }
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            this.Init_CmbImages();

            //SetNextFolder();

            //this.setCurrnetDataPaths();
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
            //HACK: some python handlers return a path, and some, returns an ID
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

        private void ImagesDir_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            this.setCurrnetDataPaths();            
        }
    }
}
