using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using System.Text.RegularExpressions;

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

        public VqaLogics(string jsonPath, string pixalMapPath, string pythonHandler)
        {
            if (String.IsNullOrWhiteSpace(jsonPath))
                throw new ArgumentException("Cannot work without a json path", nameof(jsonPath));
            this.jsonPath = jsonPath;
            this.pixalMapPath = pixalMapPath;
            this.pythonHandler = pythonHandler;
            

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
            if (data != null && !data.ContainsKey("Image Path"))
            {
                data["Image Path"] = $"'{imageName}'".Replace(@"\",@"\\");
            }
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
                string error_message = null;
                string rawData = "";
                try
                {
                    var args = $"-{option} \"{value}\"";
                    rawData = await ExecutePython(args);
                    var json_pattern = @"\{(.|\s)*\}";
                    var re = new Regex(json_pattern);
                    var match = re.Match(rawData);
                    if (match.Success)
                    {
                        //new List<int>().FirstOrDefault
                        var json = match.Groups[0].Value;
                        //var cleanJson = json.Replace("\\\"", "\"");
                        values = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);
                    }
                    else
                    {
                        error_message = String.IsNullOrWhiteSpace(rawData)?
                                "Got an Empty response":
                                $"Failed to get match for: {rawData}";
                    }
                    
                    
                }
                catch (Exception e)
                {
                    error_message = $"Got an error while quering python:\n{e}";
                    // 
                    if (!String.IsNullOrWhiteSpace(rawData) )
                        error_message += $"\n\nRaw Data:\n{rawData}";
                }

                values = values ?? new Dictionary<string, object>() { { ERROR_KEY, error_message ?? "Unknown Error"} };



            }

            if (values == null)
            {
                values = new Dictionary<string, object>
                                {
                                    { ERROR_KEY, $"Got an empty response for parameters: option: {option}; value: {value}"}
                                };
            }else if (values.Count == 0)
            {
                values = new Dictionary<string, object>
                                {
                                    { "0 length response", $"Got an response containing  no items for parameters: option: {option}; value: {value}"}
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
            Debug.Print($"Python file :\n{this.pythonHandler}");

            var processArgs = $"-p \"{this.jsonPath}\" {args}";
            Debug.Print($"Process Args:\n{processArgs}");
            var argStr = $"{this.pythonHandler} {processArgs}";
            p.StartInfo.Arguments = argStr;
            p.StartInfo.WorkingDirectory = new DirectoryInfo(Path.GetDirectoryName(this.pythonHandler)).Parent.FullName;
            p.Start();
            // Do not wait for the child process to exit before
            // reading to the end of its redirected stream.
            // p.WaitForExit();
            // Read the output stream first and then wait.
            string output = await Task.Run(() => p.StandardOutput.ReadToEnd().Trim());
            p.WaitForExit();

            if (String.IsNullOrWhiteSpace(output))
            {
                Debug.Print($"Got an empty output for \n{p.StartInfo.FileName} {p.StartInfo.Arguments}");
                Debug.WriteLine(String.Format("Args were:\n{0}",argStr));
            }
            return output;
        }
    }
}
