using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Media;
using System.Drawing.Imaging;
using VQA.Logic;
using System.IO;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using System.Windows.Navigation;
using System.Threading.Tasks;
using VQA.GUI;
using Types;
using Interfaces;

namespace SDKSamples.ImageSample
{
    public sealed partial class MainWindow : Window
    {

        public PhotoCollection Photos;
        public static List<VqaData> KnownDataLocations
        {
            get
            {
             
                var vqa2017 = new VqaData(description:"VQA 2017"
                                 ,images: @"C:\Users\Public\Documents\Data\2017\val2017"
                                 , dataFile: @"C:\Users\Public\Documents\Data\2017\annotations\stuff_val2017.json"
                                 , pixelMaps: @"C:\Users\Public\Documents\Data\2017\annotations\stuff_val2017_pixelmaps"
                                 , pythonHandler: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\parsers\VQA.Python.py");

                var vqa2014 = new VqaData(description: "VQA 2014"
                                 , images: @"C:\Users\Public\Documents\Data\2014 Train\train2014"
                                    , dataFile: @"C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json"
                                    , pixelMaps: ""
                                    , pythonHandler: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\parsers\VQA14.py");

                var vqa2015 = new VqaData(description: "VQA 2015 (2014 rev 2)"
                                 , images: vqa2014.Images
                                         , dataFile: "D:\\GitHub\\VQA-Keras-Visual-Question-Answering\\data\\Questions_Train_mscoco\\MultipleChoice_mscoco_train2014_questions.json"
                                         , pixelMaps: ""
                                         , pythonHandler: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\parsers\VQA14_multiple.py");

                var vqa2018_train = new VqaData(description: "VQA 2018 train"
                                 , images: @"C:\Users\Public\Documents\Data\2018\VQAMed2018Train\VQAMed2018Train-images"
                                 , dataFile: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\data\model_input.h5"
                                 , pixelMaps: @""
                                 , pythonHandler: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\parsers\VQA.Python.py");

                var vqa2018_validation = new VqaData(description: "VQA 2018 validation"
                                , images: @"C:\Users\Public\Documents\Data\2018\VQAMed2018Valid\VQAMed2018Valid-images"
                                , dataFile: @"C:\Users\Public\Documents\Data\2018\VQAMed2018Valid\VQAMed2018Valid-QA.csv"
                                , pixelMaps: @""
                                , pythonHandler: @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\parsers\VQA18.py");

                return new List<VqaData>{ vqa2018_train, vqa2018_validation, vqa2015, vqa2017, vqa2014};
            }
        }

        internal UIElementGenerator UiElementGenerator { get; }
        public ViewModel _viewModel => this.DataContext as ViewModel;
            
        private VqaLogics logics;

        public MainWindow()
        {
            InitializeComponent();
            this.UiElementGenerator = new UIElementGenerator(this);
            this.DataContext = new ViewModel();
            
            
        }

        private void LvModels_TargetUpdated(object sender, DataTransferEventArgs e)
        {
            throw new NotImplementedException();
        }

        private void Init_CmbImages()
        {
            var data_locations = KnownDataLocations;
            this.ImagesDir.ItemsSource = data_locations;
            var dummy = new VqaData("","", "", "", "");
            // this.ImagesDir.DisplayMemberPath = nameof(dummy.Images);
            this.ImagesDir.SelectedIndex = 0;
        }

        private void OnPhotoClick(object sender, RoutedEventArgs e)
        {
            PhotoView pvWindow = new PhotoView();
            if (lstPhotos.SelectedItem is Photo photo)
            {
                pvWindow.SelectedPhoto = photo;
                pvWindow.Show();
            }
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

        private async Task setCurrnetDataPaths()
        {
            if (this.ImagesDir.SelectedItem is VqaData vqaData)
            {
                this.logics = new VqaLogics(vqaData.Captions, vqaData.PixelMaps, vqaData.PythonHandler);

                var models = await this.logics.GetModels();
                this._viewModel.ModelsList.Clear();

                foreach (var m in models)
                    this._viewModel.ModelsList.Add(m);


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

            this.txbResponce.Text = String.Empty;
            bool query = this.rdbQuery.IsChecked ?? false;
            if (query)
                await this.QueryImaeghs();
            else
                await this.Predict();
           

        }

        private async Task Predict()
        {
            const int MAX_WORDS_TO_TAKE = 20;
            var question = this.txbQuestion.Text;
            var imagePath = (this.lstPhotos.SelectedItem as Photo)?.Path ?? "";
            var imagesDirectory = this.ImagesDir.Text;
            var hasData = !String.IsNullOrWhiteSpace(question) && File.Exists(imagePath) ;
            string responce;
            if (!hasData)
                responce = "Got invalid data to query";

            else
            {
                var prediction = await this.logics.Predict(question, new FileInfo(imagePath));
                responce = String.Join(" ", prediction.Predictions.Take(MAX_WORDS_TO_TAKE).Select(p => p.Prediction));
                var probs = String.Join("; ", prediction.Predictions.Take(MAX_WORDS_TO_TAKE).Select(p => $"{p.Prediction} ({p.Probability})"));
                responce += "\n" + probs;

            }

            

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
            var bg = this.txbQuestion.Background;
            try
            {
                this.txbQuestion.Background = Brushes.BlueViolet;
                this.txbQuestion.IsEnabled = false;


                var match_images = await this.logics.Query(question);
                match_images.Sort();
                //HACK: some python handlers return a path, and some, returns an ID
                var isFileNames = this.Photos.Any(fName => match_images.Contains(Path.GetFileName(fName.Path)));
                var hasFileNamesWithoutExt = this.Photos.Any(fName => match_images.Contains(Path.GetFileNameWithoutExtension(fName.Path)));
                Predicate<string> pred;
                if (isFileNames)
                    pred = fn => match_images.Contains(fn);
                else if (hasFileNamesWithoutExt)
                    pred = fn => match_images.Contains(Path.GetFileNameWithoutExtension(fn));
                else
                {
                    match_images = match_images.Select(id => id.PadLeft(12, '0') + ".").ToList();
                    pred = fn => match_images.Any(m => fn.Contains(m));
                }

                this.Photos.Filter = fn => pred(fn);
            }
            finally
            {
                this.txbQuestion.Background = bg;                
                this.txbQuestion.IsEnabled = true;
            }

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
                
                var dataDict = await this.logics.GetImageData(photo.image_name);
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

        private async void ImagesDir_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            await this.setCurrnetDataPaths();            
        }

        private void ListBoxItem_RequestBringIntoView(object sender, RequestBringIntoViewEventArgs e)
        {
            e.Handled = true;
        }

        private void lvModels_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var model = lvModels.SelectedItem as IModelInfo;
            if (model == null)
                return;

            bool success = this.logics.SetModel(model.Model_Id);
        }
    }
}
