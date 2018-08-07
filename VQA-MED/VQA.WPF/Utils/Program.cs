using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Utils
{
    class Program
    {
        public static void Main()
        {

            var p = new PythonQueryProxy();

            var dataframe_path = "C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\model_input.h5".Replace(@"\", @"\\");
            var res = p.AddCommand($"get_image_data(image_name='{"0392-100X-33-350-g002.jpg"}', dataframe_path='{dataframe_path}')");
            Debug.Write(res);

            //p.AddCommand($"get_image_data(image_name='{"0392-100X-33-350-g002.jpg"}')");
            //p.AddCommand($"print('hello')");

            //var printCmd = $"print('{dataframe_path}')";
            //p.AddCommand(printCmd); 

            while (true)
            {
                Thread.Sleep(1000);
            }
            return;
            var w = new PythonPredictor();
            //w.AddCommand("print(os.getcwd())");
            w.AddCommand("print(sys.path)");

            //w.AddCommand("add(1,0)");
            //w.AddCommand("predict(2)");
            //w.AddCommand("add(3)");

            //w.AddCommand("add(3,1)");
            //w.AddCommand("add(0,5)");

            Thread.Sleep(500);
            //w.Dispose();



        }
    }
}
