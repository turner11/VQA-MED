using System;
using System.Windows;
using System.Windows.Data;
using System.Xml;
using System.Configuration;
using System.Linq;

namespace SDKSamples.ImageSample
{
    public partial class app : Application
    {
        void OnApplicationStartup(object sender, StartupEventArgs args)
        {
            MainWindow mainWindow = new MainWindow();
            mainWindow.Photos = (PhotoCollection)(this.Resources["Photos"] as ObjectDataProvider).Data;
            mainWindow.Show();

            //mainWindow.SetNextFolder();
            //mainWindow.Photos.Path = mainWindow.ImagesDir.Text;

            //var captionFile = SDKSamples.ImageSample.MainWindow.KnownDataLocations.Where(tpl => String.Equals(tpl.Images, mainWindow.ImagesDir.Text, StringComparison.InvariantCulture)).FirstOrDefault().Captions ?? "";
            //mainWindow.Photos.Captions = captionFile;
        }
    }
}