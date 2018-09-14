using Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Types
{
    public class ModelInfo : IModelInfo
    {
        public int Model_Id { get; }
        public string Loss_Function { get; }
        public string Activation { get; }
        public int Trainable_Parameter_Count { get; }

        public double Bleu { get; }
        public double Wbss { get; }
        public string Notes { get; }

        public ModelInfo(int model_id, string loss_function, string activation, int trainable_parameter_count, double bleu, double wbss, string notes)
        {
            this.Model_Id = model_id;
            this.Loss_Function = loss_function;
            this.Activation = activation;
            this.Trainable_Parameter_Count = trainable_parameter_count;

            this.Bleu = bleu;
            this.Wbss = wbss;

            this.Notes = notes;
        }

        public override string ToString()
        {
            return $"Model: {Model_Id}; Loss: {Loss_Function}; Activation: {Activation}; Bleu: {Bleu}; WBSS: {Wbss}; Prarmeters: {Trainable_Parameter_Count}";
        }


    }
}
