using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace VQA.Logic
{
    public class VqaLogics
    {
        public static string Ask(string question, FileInfo imagePath)
        {
            return question;
        }

        public static async Task<string> GetImageCaptions(string jsonPath, string imageName)
        {
            if (String.IsNullOrWhiteSpace(jsonPath) || String.IsNullOrWhiteSpace(imageName))
            {
                return "";
            }
            // Start the child process.
            Process p = new Process();
            // Redirect the output stream of the child process.
            p.StartInfo.UseShellExecute = false;
            //p.StartInfo.UseShellExecute = true;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.FileName = "python";
            p.StartInfo.CreateNoWindow = true;
            var argStr = $"C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\VQA.Python.py -p \"{jsonPath}\" -n \"{imageName}\"";
            p.StartInfo.Arguments = argStr;
            p.Start();
            // Do not wait for the child process to exit before
            // reading to the end of its redirected stream.
            // p.WaitForExit();
            // Read the output stream first and then wait.
            string output = await Task.Run(()=> p.StandardOutput.ReadToEnd().Trim());
            p.WaitForExit();
            1.ToString();

            return output;
            
         
        }

    }
}
