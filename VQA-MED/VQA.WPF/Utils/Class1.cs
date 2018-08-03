using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public static class Extensions
{
    public static string GetCommandLine(this Process process)
    {
        var commandLine = new StringBuilder(process.MainModule.FileName);

        commandLine.Append(" ");
        //using (var searcher = new ManagementObjectSearcher("SELECT CommandLine FROM Win32_Process WHERE ProcessId = " + process.Id))
        //{
        //    foreach (var @object in searcher.Get())
        //    {
        //        commandLine.Append(@object["CommandLine"]);
        //        commandLine.Append(" ");
        //    }
        //}

        return commandLine.ToString();
    }
}

