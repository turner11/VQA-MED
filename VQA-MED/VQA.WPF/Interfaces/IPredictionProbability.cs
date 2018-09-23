namespace Interfaces
{
    public interface IPredictionProbability
    {
        string Prediction { get; }
        double Probability { get; }
    }
}