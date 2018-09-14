using Interfaces;
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
using Types;

namespace Utils
{
    public class PythonModelInfo : InteractivePythonWrapper
    {
        const string SCRIPT = @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\parsers\temp_cs_glue.py";
        const string WORKINGDIR = @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python";
        private const string PYTHON_INTERP_PATH = @"C:\local\Anaconda3-4.1.1-Windows-x86_64\envs\conda_env\python.exe";
        public PythonModelInfo() : base(script: SCRIPT, interpeter: PYTHON_INTERP_PATH, workingDir: WORKINGDIR)
        {

        }

        public IEnumerable<IModelInfo> GetModels()
        {
            var models = new List<IModelInfo>();

            var command = $"get_models()";
            var responce = this.ExecutePythonCommand(command);
            start:
            try
            {
                var resDict = this.CommandToDictionay<string, Dictionary<int, object>>(command);


                IEnumerable<Dictionary<int, object>> data_dictionaries = resDict.Values.Select(p => p);
                IEnumerable<int> firstKeys = data_dictionaries.FirstOrDefault().Keys.ToList();
                List<int> keys = data_dictionaries.Aggregate(firstKeys, (ks, d) => ks.Intersect(d.Keys)).ToList();
                foreach (int k in keys)
                {
                    if (!int.TryParse(resDict["model_id"][k].ToString(), out int model_id))
                        model_id = -1;
                    string loss_function = (resDict["loss_function"][k] ?? "").ToString();
                    string activation = (resDict["activation"][k] ?? "").ToString();

                    if (!int.TryParse(resDict["trainable_parameter_count"][k].ToString(), out int trainable_parameter_count))
                        trainable_parameter_count = -1;

                    double bleu = resDict["bleu"][k] as double? ?? double.NaN;
                    double wbss = resDict["wbss"][k] as double? ?? double.NaN;
                    string notes = (resDict["notes"][k] ?? "").ToString();
                    notes = notes.Replace(@"\n", Environment.NewLine);

                    var modelInfo = new ModelInfo(model_id: model_id
                                                 , loss_function: loss_function
                                                 , activation: activation
                                                 , trainable_parameter_count: trainable_parameter_count
                                                 , bleu: bleu
                                                 , wbss: wbss
                                                 , notes: notes);

                    models.Add(modelInfo);

                }



                //foreach (var pair in test)
                //{
                //    var key = pair.Key;
                //    var val = pair.Value;                   
                //    foreach (var p in val)
                //        Console.WriteLine($"{key}:\tKey: {p.Key} ;\t Value: {p.Value} ({(p.Value ?? new object()).GetType()})");

                //}

            }
            catch (Exception ex)
            {
                Debug.WriteLine("\n\n");
                Debug.WriteLine(ex);
                1.ToString();
                throw;
            }
            //goto start;
            models = models.OrderBy(m => m.Bleu + m.Wbss).Reverse().ToList();
            return models;
        }



        public bool SetModel(int modelId)
        {
            bool success = false;
            try
            {
                var command = $"set_model({modelId})";
                var result = this.ExecutePythonCommand(command);
                success = result.ToLower() == 'true';
            }
            catch (Exception)
            {
                success = false;
                Debug.WriteLine("Failed...");
            }
            return success;

        }
    }

}