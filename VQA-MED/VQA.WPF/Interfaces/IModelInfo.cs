namespace Interfaces
{
    public interface IModelInfo
    {
        int Model_Id { get; }
        string Activation { get; }
        string Loss_Function { get; }
        int Trainable_Parameter_Count { get; }
        double Bleu { get; }
        double Wbss { get; }
        string Notes { get; }
    }
}