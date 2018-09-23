using Interfaces;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;


namespace Types
{
    public class Prediction : IPrediction
    {
        public FileInfo ImagePath { get; }
        public string Question { get; }
        public ReadOnlyCollection<IPredictionProbability> Predictions { get; }

        public Prediction(FileInfo imagePath, string question, IEnumerable<PredictionProbability> predictions)
        {
            this.ImagePath = imagePath;
            this.Question = question;
            var cp = predictions.ToList();
            cp.Sort();
            
            this.Predictions = new ReadOnlyCollection<IPredictionProbability>(cp.Cast< IPredictionProbability>().ToList());
        }

        
    }
}