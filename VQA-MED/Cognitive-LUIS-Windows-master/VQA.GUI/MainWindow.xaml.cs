using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Drawing.Imaging;

namespace SDKSamples.ImageSample
{
    public sealed partial class MainWindow : Window
    {
        public PhotoCollection Photos;

        public MainWindow()
        {
            InitializeComponent();
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
            this.Photos.Path = ImagesDir.Text;
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            ImagesDir.Text = Environment.CurrentDirectory + "\\images";
        }

        private void ImagesDir_MouseDoubleClick(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            ImagesDir.Text = @"C:\Users\avitu\Google Drive\לימודים\תואר שני\VQA-MED\Data\2017\val2017";
            this.Photos.Path = ImagesDir.Text;

        }
    }
}