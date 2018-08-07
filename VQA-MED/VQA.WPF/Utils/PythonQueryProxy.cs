using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Utils
{
    public class PythonQueryProxy : InteractivePythonWrapper
    {
        const string SCRIPT = @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\parsers\VQA.Python.py";
        const string WORKINGDIR = @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\";
        public PythonQueryProxy() : base(SCRIPT, workingDir: WORKINGDIR)
        {
            
        }

       
    }

}
