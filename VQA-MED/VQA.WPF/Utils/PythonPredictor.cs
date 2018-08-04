using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Utils
{
    public class PythonPredictor : InteractivePythonWrapper
    {
        const string SCRIPT = @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.WPF\Utils\testPy.py";
        const string WORKINGDIR = @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\";
        public PythonPredictor() : base(SCRIPT, WORKINGDIR)
        {

        }
    }

}