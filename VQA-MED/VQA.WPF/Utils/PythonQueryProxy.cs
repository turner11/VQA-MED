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

        public PythonQueryProxy(string script,string interpeter=null) : base(script, interpeter)
        {            
        }



        public Dictionary<string,object> GetImageData(string imageName)
        {
            var command = $"get_image_data(image_name='{imageName}')";
            var retDict = this.CommandToDictionay<string, object>(command);
            return retDict;
        }

        public Dictionary<string, object> QuryData(string substring)
        {
            var command = $"query_data(query_string='{substring}')";
            var retDict = this.CommandToDictionay<string, object>(command);
            return retDict;
        }
        

        public static PythonQueryProxy Factory()
        {
            var inst = new PythonQueryProxy(SCRIPT);
            return inst;
        }

      
    }

}
