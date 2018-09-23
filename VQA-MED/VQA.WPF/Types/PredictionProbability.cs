using Interfaces;
using System;

namespace Types
{
    public class PredictionProbability : IComparable<PredictionProbability>, IPredictionProbability
    {
        public string Prediction { get; }
        public double Probability { get; }

        public PredictionProbability(string pred, double prob)
        {
            this.Prediction = pred;
            this.Probability = prob;
        }

        public int CompareTo(PredictionProbability other)
        {
            return other.Probability.CompareTo(this.Probability);
        }
    }
}