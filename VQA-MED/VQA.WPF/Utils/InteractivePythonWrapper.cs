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

using System.Reactive;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Text.RegularExpressions;
using Newtonsoft.Json;

namespace Utils
{
    public abstract class InteractivePythonWrapper : IDisposable
    {
        const string ERROR_KEY = "error";
        const string ENV_PATH_VARIABLE_NAME = "PATH";

        private readonly string _script;
        private readonly string _interpeter;
        public string _workDir { get; }


        private readonly Object _lockObject = new Object();
        private bool _exit;


        private readonly AutoResetEvent _signalGotCommandEvent = new AutoResetEvent(false);
        private readonly AutoResetEvent _signalGotResponceEvent = new AutoResetEvent(false);



        private readonly ObservableCollection<string> _commands = new ObservableCollection<string>();
        private readonly ObservableCollection<string> _responces = new ObservableCollection<string>();
        private ObservableCollection<string> _safeCommands
        {
            get
            {
                lock (this._lockObject)
                {
                    return this._commands;
                }
            }
        }

        private event EventHandler<DataReceivedEventArgs> _errorDataReceived;
        private event EventHandler<DataReceivedEventArgs> _outputDataReceived;



        public InteractivePythonWrapper(string script, string interpeter = null, string workingDir = null)
        {
            FileInfo fi = new FileInfo(script);
            if (!fi.Exists)
                throw new ArgumentException(nameof(script));

            this._script = script;
            this._workDir = workingDir ?? fi.Directory.FullName; //new DirectoryInfo(Path.GetDirectoryName(this._script)).Parent.FullName;
            this._interpeter = interpeter ?? "python"; 

            this._outputDataReceived += InteractivePythonWrapper__outputDataReceived;
            this._errorDataReceived += InteractivePythonWrapper__errorDataReceived;

            AddCurrentDirectoryToPath();

            this._exit = false;
            this._commands.CollectionChanged += this._commands_CollectionChanged;
            this._responces.CollectionChanged += this._responces_CollectionChanged;


            var ts = new ThreadStart(this.SpinUpPythonProcess);
            var th = new Thread(ts);
            th.Start();

        }

        private void _responces_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            if (e.Action == NotifyCollectionChangedAction.Add)
            {
                this._signalGotResponceEvent.Set();
            }
        }

        private void InteractivePythonWrapper__errorDataReceived(object sender, DataReceivedEventArgs e)
        {
            Debug.WriteLine($"Got Error:{e.Data }");
            this._responces.Add(e.Data);
        }

        private void InteractivePythonWrapper__outputDataReceived(object sender, DataReceivedEventArgs e)
        {
            Debug.WriteLine($"Got Output:{e.Data }");
            this._responces.Add(e.Data);
        }

        private void AddCurrentDirectoryToPath()
        {
            if (String.IsNullOrWhiteSpace(this._workDir))
                return;

            
            string pathvar = Environment.GetEnvironmentVariable(ENV_PATH_VARIABLE_NAME);
            var value = $"{this._workDir};" + pathvar;
            //var value = pathvar + $";{this._workDir}";
            var target = EnvironmentVariableTarget.User;

            try
            {
                Environment.SetEnvironmentVariable(ENV_PATH_VARIABLE_NAME, value, target);
                // var path_content= Environment.GetEnvironmentVariable(ENV_PATH_VARIABLE_NAME);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to add directory to path:\n{ex}");
            }
            

            Directory.SetCurrentDirectory(this._workDir);
            
        }

        protected virtual string ExecutePythonCommand(string cmd)
        {
            bool success = false;
            string res = "No Responce";
            try
            {
                this._responces.Clear();
                this._safeCommands.Clear();
                this._safeCommands.Add(cmd);
                this._signalGotResponceEvent.WaitOne();

                
                if (this._responces.Count > 0)
                {
                    res = this._responces[0];
                    this._responces.RemoveAt(0);
                }
                success = true;
            }
            finally
            {
                Debug.WriteLine($"{cmd } was {(success ? "SUCCESS" : "FAILURE")}");
            }
            
            return res;

        }

        protected static string ResponceToJson(string responce)
        {
            var json_pattern = @"\{(.|\s)*\}";
            var re = new Regex(json_pattern);
            var match = re.Match(responce);
            string json = match.Success ? match.Groups[0].Value : String.Empty;
            return json;
        }

        protected static Dictionary<TKey, TVal> JsonToDictionay<TKey, TVal>(string json)
        {
            if (String.IsNullOrWhiteSpace(json))
                throw new Exception("Got an Empty JSON");
                //return new Dictionary<TKey, TVal> { { ERROR_KEY, "Got an Empty JSON" } };

            Dictionary<TKey, TVal> values = null;

            var error_message = String.Empty;
            try
            {
                values = JsonConvert.DeserializeObject<Dictionary<TKey, TVal>>(json);
            }
            catch (Exception e)
            {
                error_message = $"Got an error while parsing responce:\n{e}";
                throw new Exception(error_message, e);
            }

            
            return values;
        }

        protected dynamic CommandToDynamic(string command)
        {
            var dynamics = this.CommandToDynamics(command).ToList();
            Debug.Assert(dynamics.Count == 1, $"Expected 1 prediction, got {dynamics.Count}");
            dynamic d = dynamics[0];
            return d;
        }
        protected IEnumerable<dynamic> CommandToDynamics(string command)
        {
            
            var resDict = this.CommandToDictionay<string, Dictionary<int, object>>(command);

            IEnumerable<Dictionary<int, object>> data_dictionaries = resDict.Values.Select(p => p);
            IEnumerable<int> firstKeys = data_dictionaries.FirstOrDefault().Keys.ToList();
            List<int> keys = data_dictionaries.Aggregate(firstKeys, (ks, d) => ks.Intersect(d.Keys)).ToList();


            //System.Dynamic.DynamicObject a = new System.Dynamic.DynamicObject();
            var dObjects = keys.Select((i, k) => new { Idx = i, Obj = new System.Dynamic.ExpandoObject() }).ToList();

            var properties = resDict.Keys;
            foreach (var pair in dObjects)
            {
                System.Dynamic.ExpandoObject currDynamic = pair.Obj;
                int idx = pair.Idx;

                foreach (var propertyName in properties)
                {
                    var propertyValue = resDict[propertyName][idx];
                    Debug.WriteLine($"{propertyName}: {propertyValue}");
                    ((IDictionary<string, object>)currDynamic)[propertyName] = propertyValue;
                }
            }

            return dObjects.Select(pair=> pair.Obj).ToList();
        }

        protected  Dictionary<TKey, TVal> CommandToDictionay<TKey, TVal>(string cmd)
        {
            var responce = this.ExecutePythonCommand(cmd);
            var retDict = PythonQueryProxy.ResponceToDictonary<TKey, TVal>(responce);
            return retDict;
        }


        protected static Dictionary<TKey, TVal> ResponceToDictonary<TKey, TVal>(string responce)
        {
            Dictionary<TKey, TVal> retDict = null;
            try
            {
                var json = InteractivePythonWrapper.ResponceToJson(responce);
                retDict = InteractivePythonWrapper.JsonToDictionay<TKey, TVal>(json);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to convert responce to dictionary:\n{ex.ToString()}");
                throw;
            }
            return retDict;
        }

        private void SpinUpPythonProcess()
        {
            
            Process process = new Process();
            process.StartInfo.FileName = this._interpeter;

            var path_content = Environment.GetEnvironmentVariable(ENV_PATH_VARIABLE_NAME);
            process.StartInfo.EnvironmentVariables[ENV_PATH_VARIABLE_NAME] = path_content;

            process.StartInfo.EnvironmentVariables["PYTHONPATH"] = this._workDir;
            
            // Set UseShellExecute to false for redirection, 
            //AND Required for EnvironmentVariables to be set.
            process.StartInfo.UseShellExecute = false;
            //process.StartInfo.CreateNoWindow = true;

            // Redirect the standard output of the sort command.  
            // This stream is read asynchronously using an event handler.
            process.StartInfo.RedirectStandardOutput = true;
            var sortOutput = new StringBuilder("");

            // Set our event handler to asynchronously read the sort output.
            process.OutputDataReceived += (s, e) => this._outputDataReceived?.Invoke(this, e);
            process.ErrorDataReceived += (s, e) => this._errorDataReceived?.Invoke(this, e);

            // Redirect standard input as well.  This stream is used synchronously.
            process.StartInfo.RedirectStandardInput = true;
            
            var argStr = $"-i {this._script}";
            process.StartInfo.Arguments = argStr;
            if (!String.IsNullOrWhiteSpace(this._workDir))
                process.StartInfo.WorkingDirectory = this._workDir;

            var reproduceCmds = new List<String>();
            reproduceCmds.Add($"\n\ncd {this._workDir}");
            reproduceCmds.Add($"set PYTHONPATH={this._workDir}");
            reproduceCmds.Add($"{this._interpeter} {argStr}\n");
            reproduceCmds.Add($"import sys");
            reproduceCmds.Add($"sys.path.append(r'{this._workDir}')");
            var reproduceCmd = String.Join(Environment.NewLine, reproduceCmds);
            Debug.WriteLine(reproduceCmd);

            process.Start();

            // Use a stream writer to synchronously write the sort input.
            StreamWriter sWriter = process.StandardInput;

            // Start the asynchronous read of the sort output stream.
            process.BeginOutputReadLine();
            
            while (!this._exit)
            {
                string args = this.getNextCommand();
                Debug.WriteLine($"Args: {args}");


                if (!String.IsNullOrEmpty(args))
                {
                    sWriter.WriteLine(args);
                }

                if (this._safeCommands.Count == 0)
                {
                    this._signalGotCommandEvent.WaitOne();
                }
            }
        }
        
        private string getNextCommand()
        {
            var command = "";
            if (this._safeCommands.Count > 0)
                lock (this._lockObject)
                {
                    command = this._commands[0];
                    this._commands.RemoveAt(0);
                }

            return command;
        }

        private void _commands_CollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            if (e.Action == NotifyCollectionChangedAction.Add)
                this._signalGotCommandEvent.Set();
        }

        public void Dispose()
        {
            this._exit = true;
            this._signalGotCommandEvent.Dispose();
        }


    }

}
