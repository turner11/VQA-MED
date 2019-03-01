using Interfaces;
using Newtonsoft.Json.Linq;
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
        public PythonModelInfo() : base(script: SCRIPT, interpeter: PYTHON_INTERP_PATH)
        {

        }

        public IEnumerable<IModelInfo> GetModels()
        {
            var models = new List<IModelInfo>();

            var command = $"get_models()";
            var responce = this.ExecutePythonCommand(command);
            start:
            IEnumerable<dynamic> dynamics = this.CommandToDynamics(command);


            foreach (dynamic currDynamic in dynamics)
            {
                try
                {
                    int model_id = Convert.ToInt32(currDynamic.model_id);
                    string loss_function = currDynamic.loss_function?.ToString() ?? "";
                    string activation = currDynamic.activation?.ToString() ?? "";
                    int trainable_parameter_count = Convert.ToInt32(currDynamic.trainable_parameter_count ?? -1);
                    double bleu = Convert.ToDouble(currDynamic.bleu ?? Double.NaN);
                    double wbss = Convert.ToDouble(currDynamic.wbss ?? Double.NaN);
                    string notes = currDynamic.notes?.ToString() ?? "";
                    notes = notes.Replace("\\n", Environment.NewLine);
                    

                    var modelInfo = new ModelInfo(model_id: model_id
                                                 , loss_function: loss_function
                                                 , activation: activation
                                                 , trainable_parameter_count: trainable_parameter_count
                                                 , bleu: bleu
                                                 , wbss: wbss
                                                 , notes: notes);
                    
                    models.Add(modelInfo);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine("\n\n");
                    Debug.WriteLine(ex);
                    1.ToString();
                    throw;
                }

            }
            //goto start;
            models = models.OrderBy(m => m.Bleu + m.Wbss).Reverse().ToList();
            return models;
        }

        public bool SetModel(int modelId)
        {
            bool success = false;
            var command = $"set_model({modelId})";
            try
            {
                var result = this.ExecutePythonCommand(command);
                success = result.ToLower() == "true";
            }
            catch (Exception)
            {
                success = false;
            }
            return success;

        }

        public Prediction Predict(string question, FileInfo imagePath)
        {
            bool success = false;
            Prediction result = null;
            var command = $"predict(question='{question}',image_path=r'{imagePath}')";
            var dynamicObj = this.CommandToDynamic(command);

            start:
            try
            {
                string image_name = dynamicObj.image_name;
                string path = dynamicObj?.path ?? "";
                var returned_path = new FileInfo(path);
                string q = dynamicObj.question;

                Debug.Assert(question == q, "Unexpectedly got a different question then expected.");
                Debug.Assert(returned_path.FullName == imagePath.FullName, "Unexpectedly got a different question then expected.");
                string rawPrediction = dynamicObj.prediction;
                var cleanPredictions = rawPrediction.Split(' ');

                var probText = dynamicObj.probabilities.ToString();
                var rawProbs = (List<string>)Newtonsoft.Json.JsonConvert.DeserializeObject<List<string>>(probText);
                var cleanProbs = rawProbs[0].Replace("(", String.Empty).Replace(")", String.Empty)
                                    .Split(',')
                                    .Select(v => Convert.ToDouble(v)).ToList();

                var predictions = cleanPredictions.Zip(cleanProbs, (pred, prob) => new PredictionProbability(pred, prob));
                result = new Prediction(imagePath, question, predictions);



            }
            catch (Exception ex)
            {
                Debug.WriteLine("\n\n");
                Debug.WriteLine(ex);
                1.ToString();

                result = new Prediction(imagePath, ex.Message, new List<PredictionProbability>());
            }
            
            //goto start;

            return result;
        }
    }

}