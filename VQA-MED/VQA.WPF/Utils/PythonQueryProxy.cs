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
        static readonly string DATAFRAME_PATH = "C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\model_input.h5".Replace(@"\", @"\\");
        const string SCRIPT = @"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\parsers\VQA.Python.py";
        private const string PYTHON_INTERP_PATH = @"C:\local\Anaconda3-4.1.1-Windows-x86_64\envs\conda_env\python.exe";


        const string WORKINGDIR = null;//@"C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\VQA.Python\";
        private readonly string dataFramePath;

        public PythonQueryProxy(string script, string dataFramePath, string interpeter=null) : base(script, interpeter)
        {
            this.dataFramePath = dataFramePath;
        }



        public Dictionary<string,object> GetImageData(string imageName)
        {
            var command = $"get_image_data(image_name='{imageName}', dataframe_path='{this.dataFramePath}')";
            var retDict = this.CommandToDictionay(command);
            return retDict;
        }

        public static PythonQueryProxy Factory()
        {
            var inst = new PythonQueryProxy(SCRIPT, DATAFRAME_PATH, PYTHON_INTERP_PATH);
            return inst;
        }
    }

}
