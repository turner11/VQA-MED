using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Linq;

namespace VQA.Logic
{
    public class VqaLogics
    {
        private const string PYTHON_INTERP_PATH = "\"C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Anaconda3_64\\python.exe\"";
        public const string ERROR_KEY = "error"; //this is also impelemnted on python's side

        public readonly string jsonPath;
        public readonly string pixalMapPath;

        public VqaLogics(string jsonPath, string pixalMapPath)
        {
            if (String.IsNullOrWhiteSpace(jsonPath))
                throw new ArgumentException("Cannot work without a json path", nameof(jsonPath));
            this.jsonPath = jsonPath;
            this.pixalMapPath = pixalMapPath;

        }
        public async Task<string> Ask(string question, FileInfo imagePath)
        {
            var data = await this.QueryPython("a", question); 
            return String.Join("; ",data.Select(pair => pair.Value));
        }


        public async Task<List<string>> Query(string question)
        {
            
            var query = question;
            var data = await this.QueryPython("q", query);


            var match_images = data.Select(pair =>  pair.Key).ToList();
            return match_images;
            
        }

        public async Task<string> GetImageCaptions(string imageName)
        {
            var data = await this.GetImageData(imageName);
            data.TryGetValue("caption", out object caption);
            return caption.ToString();
        }

        public async Task<Dictionary<string, object>> GetImageData(string imageName)
        {
            var data =  await this.QueryPython("n", imageName);
            var fi = new FileInfo(imageName);
            var pixelMapImage = new FileInfo(Path.Combine(this.pixalMapPath, fi.Name).ToLower().Replace(".jpg",".png"));
            if (pixelMapImage.Exists)
            {
                data["Pixel Map"] = pixelMapImage.FullName;
            }
            return data;
           
        }

        public async Task<Dictionary<string, object>> QueryPython(string option, string value)
        {
            Dictionary<string, object>  values = null;
            if (!String.IsNullOrWhiteSpace(option) && !String.IsNullOrWhiteSpace(value))
            {
                try
                {
                    var args = $"-{option} \"{value}\"";
                    var rawData = await ExecutePython(args);
                    values = JsonConvert.DeserializeObject<Dictionary<string, object>>(rawData);
                    //var valuesAAAA = JsonConvert.DeserializeObject<Dictionary<string, string>>(rawData);
                }
                catch (Exception e)
                {

                    values = new Dictionary<string, object>() { { ERROR_KEY, $"Got an error while quering python:\n{e}" } };
                }

               

            }

            if (values == null || values.Count == 0)
            {
                values = new Dictionary<string, object>
                                {
                                    { ERROR_KEY, $"Got an empty response for parameters: option: {option}; value: {value}"}
                                };
            }

            return values;


        }

        private async Task<string> ExecutePython(string args)
        {
            // Start the child process.
            Process p = new Process();
            // Redirect the output stream of the child process.
            p.StartInfo.UseShellExecute = false;
            //p.StartInfo.UseShellExecute = true;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.FileName = PYTHON_INTERP_PATH;
            p.StartInfo.CreateNoWindow = true;
            Debug.Print($"JSON path:\n{this.jsonPath}");
            Debug.Print($"args:\n{args}");
            var argStr = $"C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\VQA.Python.py -p \"{this.jsonPath}\" {args}";
            p.StartInfo.Arguments = argStr;
            p.Start();
            // Do not wait for the child process to exit before
            // reading to the end of its redirected stream.
            // p.WaitForExit();
            // Read the output stream first and then wait.
            string output = await Task.Run(() => p.StandardOutput.ReadToEnd().Trim());
            p.WaitForExit();

            if (String.IsNullOrWhiteSpace(output))
            {
                Debug.WriteLine(String.Format("Got an ampty responce for:\n{0}",argStr));
            }
            return output;
        }
    }
}
