using Interfaces;
using System.Collections.ObjectModel;
using System.ComponentModel;

namespace SDKSamples.ImageSample
{
    public class ViewModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        public ObservableCollection<IModelInfo> ModelsList { get; }

        private IModelInfo _selectedModel;
        public string ModelImagePath => this.SelectedModel?.ImagePath;
        public string ModelSummary => this.SelectedModel?.Summary;

        public IModelInfo SelectedModel
        {
            get { return this._selectedModel; }
            set { this._selectedModel = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(SelectedModel)));
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ModelImagePath)));
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ModelSummary)));
            }
        }

        

        public ViewModel()
        {
            this.ModelsList = new ObservableCollection<IModelInfo>();
        }

        
    }
}