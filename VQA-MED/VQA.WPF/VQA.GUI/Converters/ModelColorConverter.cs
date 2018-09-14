using Interfaces;
using System;
using System.Collections.Generic;

using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using System.Windows.Media;

namespace VQA.GUI.Converters
{
    class ModelColorConverter : IValueConverter
    {
        const double SCALING_FACTOR = 2;

        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var modelInfo = value as IModelInfo;
            if (modelInfo == null)
                return null;

            var grad = modelInfo.Wbss + modelInfo.Bleu;

            var color = Colors.Green;
            color = GetColorFromRedYellowGreenGradient(grad);

            if (modelInfo.Wbss < 0.10)
                color = Colors.Red;

            var brush = new SolidColorBrush(color);
            return brush;
            
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }

        static Color GetColorFromRedYellowGreenGradient(double percentage)
        {
            var val = (float)(percentage * SCALING_FACTOR);
            if (val > 1)
                return Colors.Green;
            
            var red = 1.0f - val;
            var green = 1.0f - red;
            var blue = 0.0f;

            //Color resulta = Color.FromValues(new float[] { (float)red, (float)green, blue }, new Uri("www.dummy.com"));
            Color result = Color.FromScRgb(0.5f, red, green, blue);
            return result;
        }

    }
}
