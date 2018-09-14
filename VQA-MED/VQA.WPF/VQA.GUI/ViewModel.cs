using Interfaces;
using System.Collections.ObjectModel;

namespace SDKSamples.ImageSample
{
    public class ViewModel
    {
        
        public ObservableCollection<IModelInfo> ModelsList { get; }

        public ViewModel()
        {
            this.ModelsList = new ObservableCollection<IModelInfo>();
        }

    }
}