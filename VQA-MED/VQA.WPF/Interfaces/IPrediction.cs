using System.Collections.ObjectModel;
using System.IO;

namespace Interfaces
{
    public interface IPrediction
    {
        FileInfo ImagePath { get; }
        ReadOnlyCollection<IPredictionProbability> Predictions { get; }
        string Question { get; }
    }
}