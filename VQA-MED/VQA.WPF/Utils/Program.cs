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
            start:
            PythonQueryProxy p = PythonQueryProxy.Factory();

            string imageName = "0392-100X-33-350-g002.jpg";
            var res = p.GetImageData(imageName);
            

            Debug.WriteLine("------------------------------------------------------");
            foreach (var pair in res)
                Debug.WriteLine("{0}:\n\n{1}", pair.Key, pair.Value);
            Debug.WriteLine("------------------------------------------------------");

            
            p.Dispose();
            goto start;
            return;

        }
    }
}
