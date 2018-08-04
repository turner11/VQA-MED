using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Utils
{
    public class InteractivePythonWrapper:IDisposable
    {
        private const string PYTHON_INTERP_PATH = @"C:\local\Anaconda3-4.1.1-Windows-x86_64\envs\conda_env\python.exe";

        private readonly string _script;
        private readonly string _interpeter;
        public string _workDir { get; }

        
        private readonly Object _lockObject = new Object();
        private bool _exit;


        private readonly AutoResetEvent _signalEvent = new AutoResetEvent(false);
        
        private readonly ObservableCollection<string> _commands = new ObservableCollection<string>();
        private ObservableCollection<string> safeCommands {
            get
            {
                lock (this._lockObject)
                {
                    return this._commands;
                }
            }
        }

        public event EventHandler<string> GotResults;
        public event EventHandler<string> GotError;




        public InteractivePythonWrapper(string script, string interpeter=null, string workingDir=null)
        {
            FileInfo fi = new FileInfo(script);
            if (!fi.Exists)
                throw new ArgumentException(nameof(script));

            this._script = script;
            this._workDir = workingDir ?? fi.Directory.FullName;
            this._interpeter = interpeter ?? PYTHON_INTERP_PATH;
            //new DirectoryInfo(Path.GetDirectoryName(this._script)).Parent.FullName;

            AddCurrentDirectoryToPath();

            this._exit = false;
            this._commands.CollectionChanged += this._commands_CollectionChanged;


            var ts = new ThreadStart(this.SpinUpPythonProcess);
            var th = new Thread(ts);
            th.Start();

        }

        private void AddCurrentDirectoryToPath()
        {
            if (String.IsNullOrWhiteSpace(this._workDir))
                return;

            const string name = "PATH";
            string pathvar = Environment.GetEnvironmentVariable(name);
            var value = pathvar + $";{this._workDir}";
            var target = EnvironmentVariableTarget.Machine;
            Environment.SetEnvironmentVariable(name, value, target);
            Environment.SetEnvironmentVariable("PATH", value, target);

            Directory.SetCurrentDirectory(this._workDir);
        }

        public void AddCommand(string cmd)
        {
            this.safeCommands.Add(cmd);
        }

        private void SpinUpPythonProcess()
        {

            Process process = new Process();
            process.StartInfo.FileName = this._interpeter;

            // Set UseShellExecute to false for redirection.
            process.StartInfo.UseShellExecute = false;
            //process.StartInfo.CreateNoWindow = true;

            // Redirect the standard output of the sort command.  
            // This stream is read asynchronously using an event handler.
            process.StartInfo.RedirectStandardOutput = true;
            var sortOutput = new StringBuilder("");

            // Set our event handler to asynchronously read the sort output.
            process.OutputDataReceived += new DataReceivedEventHandler(this.gotOutPutHandler);
            process.ErrorDataReceived += new DataReceivedEventHandler(this.errorOutPutHandler);
            


            // Redirect standard input as well.  This stream
            // is used synchronously.
            process.StartInfo.RedirectStandardInput = true;

            //Debug.Print($"args:\n{args}");
            //var processArgs = $"-p \"{this.jsonPath}\" {args}";
            //Debug.Print($"Process Args:\n{processArgs}");
            var argStr = $"-i {this._script}";
            process.StartInfo.Arguments = argStr;
            if (!String.IsNullOrWhiteSpace(this._workDir))
            process.StartInfo.WorkingDirectory = this._workDir;
            
            process.Start();

            // Use a stream writer to synchronously write the sort input.
            StreamWriter sWriter = process.StandardInput;

            // Start the asynchronous read of the sort output stream.
            process.BeginOutputReadLine();
            
            String inputText;
            int numInputLines = 0;
            while(!this._exit)
            {
                string args = this.getNextCommand();
                Debug.WriteLine($"Args: {args}");

                
                if (!String.IsNullOrEmpty(args))
                {
                    numInputLines++;
                    sWriter.WriteLine(args);
                }

                //inputText = Console.ReadLine();

                if (this.safeCommands.Count == 0)
                {
                    this._signalEvent.WaitOne();
                }
            }

            if (false)
            {

                // Start the child process.
                Process p = new Process();
                // Redirect the output stream of the child process.
                p.StartInfo.UseShellExecute = false;
                //p.StartInfo.UseShellExecute = true;
                p.StartInfo.RedirectStandardOutput = true;
                p.StartInfo.FileName = PYTHON_INTERP_PATH;
                p.StartInfo.CreateNoWindow = true;

                //Debug.Print($"args:\n{args}");



                //var processArgs = $"-p \"{this.jsonPath}\" {args}";
                //Debug.Print($"Process Args:\n{processArgs}");


                var argStra = $"-i ";
                p.StartInfo.Arguments = argStr;
                p.StartInfo.WorkingDirectory = new DirectoryInfo(Path.GetDirectoryName(this._script)).Parent.FullName;
                p.Start();
                // Do not wait for the child process to exit before
                // reading to the end of its redirected stream.
                // p.WaitForExit();
                // Read the output stream first and then wait.
                string output = p.StandardOutput.ReadToEnd().Trim();

                p.WaitForExit();

                if (String.IsNullOrWhiteSpace(output))
                {
                    Debug.Print($"Got an empty output for \n{p.StartInfo.FileName} {p.StartInfo.Arguments}");
                    Debug.WriteLine(String.Format("Args were:\n{0}", argStr));
                }
                //return output;
            }
        }



        

        private string getNextCommand()
        {
            var command = "";
            if (this.safeCommands.Count > 0)
                lock (this._lockObject)
                {
                    command= this._commands[0];
                    this._commands.RemoveAt(0);
                }

            return command;
        }

        private void _commands_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            this._signalEvent.Set();
        }

        private void errorOutPutHandler(object sender, DataReceivedEventArgs e)
        {
            Debug.WriteLine($"@@@@@@ Error: {e.Data}");
        }
        private void gotOutPutHandler(object sender, DataReceivedEventArgs e)
        {
            Debug.WriteLine($"@@@@@@ Data: {e.Data}");
        }

        public void Dispose()
        {
            this._exit = true;
            this._signalEvent.Dispose();
        }
    }
    

    

}