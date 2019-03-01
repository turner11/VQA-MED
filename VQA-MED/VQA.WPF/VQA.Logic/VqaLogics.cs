using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using System.Text.RegularExpressions;
using Utils;
using Interfaces;

namespace VQA.Logic
{
    public class VqaLogics
    {
        public const string ERROR_KEY = "error"; //this is also impelemnted on python's side

        public readonly string jsonPath;
        public readonly string pixalMapPath;
        public readonly string pythonHandler;
        private PythonQueryProxy _pythonProxy;
        private PythonModelInfo _pythonModelProxy;

        public VqaLogics(string jsonPath, string pixalMapPath, string pythonHandler)
        {
            //if (String.IsNullOrWhiteSpace(jsonPath))
            //    throw new ArgumentException("Cannot work without a json path", nameof(jsonPath));
            this.jsonPath = jsonPath;
            this.pixalMapPath = pixalMapPath;
            this.pythonHandler = pythonHandler;
            this._pythonProxy = PythonQueryProxy.Factory();
            this._pythonModelProxy = new PythonModelInfo();
        }
        public async Task<IPrediction> Predict(string question, FileInfo imagePath)
        {
            var result = await Task.Run(()=>this._pythonModelProxy.Predict(question, imagePath));
            return result;
        }


        public async Task<List<string>> Query(string substrig)
        {
            if (substrig.ToLower() == "reset".ToLower())
            {
                
                var old = this._pythonProxy;
                var old_model = this._pythonModelProxy;
                this._pythonProxy = PythonQueryProxy.Factory();
                this._pythonModelProxy = new PythonModelInfo();

                old.Dispose();
                old_model.Dispose();
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
                a.ToString();
                try
                {
                    var b= JsonConvert.DeserializeObject<object>(a.ToString());
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
            var data = await Task.Run(() => this._pythonProxy?.GetImageData(imageName));

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

        public async Task<IEnumerable<IModelInfo>> GetModels()
        {
            var ms = await Task.Run(() =>this._pythonModelProxy.GetModels());
            //var filter = new int[] {77 ,98 ,85 ,93, 95 ,79 ,89 ,87 ,83 ,162};
            //ms = ms.Where(m => filter.Contains(m.Model_Id)).ToList();


            return ms;
        }

        public bool SetModel(int modelId)
        {
            return this._pythonModelProxy.SetModel(modelId);
        }
    }
}