using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using System.Text.RegularExpressions;
using Utils;

namespace VQA.Logic
{
    public class VqaLogics
    {
        private const string PYTHON_INTERP_PATH = @"C:\local\Anaconda3-4.1.1-Windows-x86_64\envs\conda_env\python.exe";
        //"\"C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Anaconda3_64\\python.exe\"";
        //"\"C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Anaconda3_64\\python.exe\"";
        public const string ERROR_KEY = "error"; //this is also impelemnted on python's side

        public readonly string jsonPath;
        public readonly string pixalMapPath;
        public readonly string pythonHandler;
        private PythonQueryProxy _pythonProxy;

        public VqaLogics(string jsonPath, string pixalMapPath, string pythonHandler)
        {
            if (String.IsNullOrWhiteSpace(jsonPath))
                throw new ArgumentException("Cannot work without a json path", nameof(jsonPath));
            this.jsonPath = jsonPath;
            this.pixalMapPath = pixalMapPath;
            this.pythonHandler = pythonHandler;
            this._pythonProxy = PythonQueryProxy.Factory();
        }
        public async Task<string> Ask(string question, FileInfo imagePath)
        {
            throw new NotImplementedException();
        }


        public async Task<List<string>> Query(string substrig)
        {
            if (substrig.ToLower() == "reset".ToLower())
            {
                var old = this._pythonProxy;
                this._pythonProxy = PythonQueryProxy.Factory();
                old.Dispose();
                return new List<string>();
            }
            var data = await Task.Run(() => this._pythonProxy.QuryData(substrig));
            DEBUG(data);
            return null;// caption.ToString();


            //var match_images = data.Select(pair =>  pair.Key).ToList();            
            //return match_images;

        }

        private static void DEBUG(object obj)

        {
            var data = obj as Dictionary<string, object>;
            data.TryGetValue("image_name", out object caption);
            if (caption is IEnumerable<object> en2)
            {
                var a = en2.FirstOrDefault();
                try
                {
                    JsonConvert.DeserializeObject<string>(a.ToString());
                }
                catch (Exception e)
                {
                    Debug.WriteLine(e);
                }
            }
        }

        public async Task<string> GetImageCaptions(string imageName)
        {
            var data = await this.GetImageData(imageName);
            data.TryGetValue("caption", out object caption);
            return caption.ToString();
        }

        public async Task<Dictionary<string, object>> GetImageData(string imageName)
        {
            var data = await Task.Run(() => this._pythonProxy.GetImageData(imageName));

            if (data != null && !data.ContainsKey("Image Path"))
            {
                data["Image Path"] = $"'{imageName}'".Replace(@"\", @"\\");
            }
            var fi = new FileInfo(imageName);
            var pixelMapImage = new FileInfo(Path.Combine(this.pixalMapPath, fi.Name).ToLower().Replace(".jpg", ".png"));
            if (pixelMapImage.Exists)
            {
                data["Pixel Map"] = pixelMapImage.FullName;
            }
            return data;

        }




    }
}
