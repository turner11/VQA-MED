using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows.Media.Imaging;
using System.Linq;

namespace SDKSamples.ImageSample
{
    /// <summary>
    /// This class describes a single photo - its location, the image and 
    /// the metadata extracted from the image.
    /// </summary>
    public class Photo
    {
        public Photo(string path)
        {
            this._path = path;
            this._source = null;
            this._image = null;
            this._metadata = null;
            this.image_name = new FileInfo(path).Name;
        }

        public override string ToString()
        {
            return Source.ToString();
        }

        private string _path;
        public string Path { get { return _path; } }

        private Uri _source;
        public Uri Source
        {
            get
            {
                if (this._source == null)
                    this._source = new Uri(this.Path); 
                return this._source;
            }
        }


    private BitmapFrame _image;
        public BitmapFrame Image
        {
            get
            {
                if (this._image == null)
                    try
                    {
                        this._image = BitmapFrame.Create(Source);
                    }
                    catch (ArgumentException ex)
                    {
                        Debug.WriteLine($"Failed to create image for '{this.Path}'");
                        this._image = null;
                    }
                return _image;
            }
            set { _image = value; }
        }

        private ExifMetadata _metadata;

        public readonly string image_name;

        public ExifMetadata Metadata
        {
            get
            {
                this._metadata = this._metadata ?? new ExifMetadata(Source);
                return _metadata;
            }
        }

    }

    /// <summary>
    /// This class represents a collection of photos in a directory.
    /// </summary>
    public class PhotoCollection : ObservableCollection<Photo>
    {
        public string PixelMaps { get; internal set; }

        private Predicate<string> _filter;
        public Predicate<string> Filter
        {
            get { return this._filter; }
            set
            {
                this._filter = value;
                this.Update();
            }
        }

        public string Path
        {
            set
            {
                _directory = new DirectoryInfo(value);
                Update();
            }
            get { return _directory.FullName; }
        }

        private FileInfo _captions;

        public string Captions
        {
            get { return _captions?.FullName ?? ""; }
            set { _captions = new FileInfo(value); }
        }

        public DirectoryInfo Directory
        {
            set
            {
                _directory = value;
                Update();
            }
            get { return _directory; }
        }


        public PhotoCollection() { }

        public PhotoCollection(string path) : this(new DirectoryInfo(path)) { }

        public PhotoCollection(DirectoryInfo directory)
        {
            _directory = directory;
            Update();
        }

      
        private void Update()
        {
            this.Clear();
            try
            {
                var allFiles = _directory.GetFiles("*.jpg");
                var filesToAdd = allFiles.Where(f => this.Filter == null || this.Filter(f.Name)).Select(f => f.FullName).ToList();
                filesToAdd.Sort();
                foreach (var f in filesToAdd)
                {
                    Add(new Photo(f));
                }

            }
            catch (DirectoryNotFoundException)
            {
                System.Windows.MessageBox.Show($"No Such Directory: {this._directory}");
            }
        }

        DirectoryInfo _directory;
    }


    public enum ColorRepresentation
    {
        sRGB,
        Uncalibrated
    }

    public enum FlashMode
    {
        FlashFired,
        FlashDidNotFire
    }

    public enum ExposureMode
    {
        Manual,
        NormalProgram,
        AperturePriority,
        ShutterPriority,
        LowSpeedMode,
        HighSpeedMode,
        PortraitMode,
        LandscapeMode,
        Unknown
    }

    public enum WhiteBalanceMode
    {
        Daylight,
        Fluorescent,
        Tungsten,
        Flash,
        StandardLightA,
        StandardLightB,
        StandardLightC,
        D55,
        D65,
        D75,
        Other,
        Unknown
    }

    public class ExifMetadata
    {
        BitmapMetadata _metadata;

        public ExifMetadata(Uri imageUri)
        {
            try
            {
                BitmapFrame frame = BitmapFrame.Create(imageUri, BitmapCreateOptions.DelayCreation, BitmapCacheOption.None);
                _metadata = (BitmapMetadata)frame.Metadata;
            }
            catch (ArgumentException ex)
            {
                Debug.WriteLine($"Failed to create meta data for '{imageUri.AbsolutePath}'");
                
            }
            

        }

        private decimal ParseUnsignedRational(ulong exifValue)
        {
            return (decimal)(exifValue & 0xFFFFFFFFL) / (decimal)((exifValue & 0xFFFFFFFF00000000L) >> 32);
        }

        private decimal ParseSignedRational(long exifValue)
        {
            return (decimal)(exifValue & 0xFFFFFFFFL) / (decimal)((exifValue & 0x7FFFFFFF00000000L) >> 32);
        }

        private object QueryMetadata(string query)
        {
            if (_metadata == null || !_metadata.ContainsQuery(query))
                return null;
            
            return _metadata.GetQuery(query);
            
        }

        public uint? Width
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif/subifd:{uint=40962}");
                if (val == null)
                {
                    return null;
                }
                else
                {
                    if (val.GetType() == typeof(UInt32))
                    {
                        return (uint?)val;
                    }
                    else
                    {
                        return System.Convert.ToUInt32(val);
                    }
                }
            }
        }

        public uint? Height
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif/subifd:{uint=40963}");
                if (val == null)
                {
                    return null;
                }
                else
                {
                    if (val.GetType() == typeof(UInt32))
                    {
                        return (uint?)val;
                    }
                    else
                    {
                        return System.Convert.ToUInt32(val);
                    }
                }
            }
        }

        public decimal? HorizontalResolution
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif:{uint=282}");
                if (val != null)
                {
                    return ParseUnsignedRational((ulong)val);
                }
                else
                {
                    return null;
                }
            }
        }

        public decimal? VerticalResolution
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif:{uint=283}");
                if (val != null)
                {
                    return ParseUnsignedRational((ulong)val);
                }
                else
                {
                    return null;
                }
            }
        }

        public string EquipmentManufacturer
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif:{uint=271}");
                return (val != null ? (string)val : String.Empty);
            }
        }

        public string CameraModel
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif:{uint=272}");
                return (val != null ? (string)val : String.Empty);
            }
        }

        public string CreationSoftware
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif:{uint=305}");
                return (val != null ? (string)val : String.Empty);
            }
        }

        public ColorRepresentation ColorRepresentation
        {
            get
            {
                var md = QueryMetadata("/app1/ifd/exif/subifd:{uint=40961}") ?? (ushort)0;
                if ((ushort)md == 1)
                    return ColorRepresentation.sRGB;
                else
                    return ColorRepresentation.Uncalibrated;
            }
        }

        public decimal? ExposureTime
        {
            get
            {
                var val = QueryMetadata("/app1/ifd/exif/subifd:{uint=33434}") as ulong?;
                if (val.HasValue)
                {
                    return ParseUnsignedRational(val.Value);
                }
                else
                {
                    return null;
                }
            }
        }

        public decimal? ExposureCompensation
        {
            get
            {
                var val = QueryMetadata("/app1/ifd/exif/subifd:{uint=37380}") as long?;
                if (val.HasValue)
                {
                    return ParseSignedRational(val.Value);
                }
                else
                {
                    return null;
                }
            }
        }

        public decimal? LensAperture
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif/subifd:{uint=33437}");
                if (val != null)
                {
                    return ParseUnsignedRational((ulong)val);
                }
                else
                {
                    return null;
                }
            }
        }

        public decimal? FocalLength
        {
            get
            {
                var val = QueryMetadata("/app1/ifd/exif/subifd:{uint=37386}") as ulong?;
                if (val.HasValue)
                {
                    return ParseUnsignedRational(val.Value);
                }
                else
                {
                    return null;
                }
            }
        }

        public ushort? IsoSpeed
        {
            get
            {
                return (ushort?)QueryMetadata("/app1/ifd/exif/subifd:{uint=34855}");
            }
        }

        public FlashMode FlashMode
        {
            get
            {
                if ((ushort)QueryMetadata("/app1/ifd/exif/subifd:{uint=37385}") % 2 == 1)
                    return FlashMode.FlashFired;
                else
                    return FlashMode.FlashDidNotFire;
            }
        }

        public ExposureMode ExposureMode
        {
            get
            {
                ushort? mode = (ushort?)QueryMetadata("/app1/ifd/exif/subifd:{uint=34850}");

                if (mode == null)
                {
                    return ExposureMode.Unknown;
                }
                else
                {
                    switch ((int)mode)
                    {
                        case 1: return ExposureMode.Manual;
                        case 2: return ExposureMode.NormalProgram;
                        case 3: return ExposureMode.AperturePriority;
                        case 4: return ExposureMode.ShutterPriority;
                        case 5: return ExposureMode.LowSpeedMode;
                        case 6: return ExposureMode.HighSpeedMode;
                        case 7: return ExposureMode.PortraitMode;
                        case 8: return ExposureMode.LandscapeMode;
                        default: return ExposureMode.Unknown;
                    }
                }
            }
        }

        public WhiteBalanceMode WhiteBalanceMode
        {
            get
            {
                ushort? mode = (ushort?)QueryMetadata("/app1/ifd/exif/subifd:{uint=37384}");

                if (mode == null)
                {
                    return WhiteBalanceMode.Unknown;
                }
                else
                {
                    switch ((int)mode)
                    {
                        case 1: return WhiteBalanceMode.Daylight;
                        case 2: return WhiteBalanceMode.Fluorescent;
                        case 3: return WhiteBalanceMode.Tungsten;
                        case 10: return WhiteBalanceMode.Flash;
                        case 17: return WhiteBalanceMode.StandardLightA;
                        case 18: return WhiteBalanceMode.StandardLightB;
                        case 19: return WhiteBalanceMode.StandardLightC;
                        case 20: return WhiteBalanceMode.D55;
                        case 21: return WhiteBalanceMode.D65;
                        case 22: return WhiteBalanceMode.D75;
                        case 255: return WhiteBalanceMode.Other;
                        default: return WhiteBalanceMode.Unknown;
                    }
                }
            }
        }

        public DateTime? DateImageTaken
        {
            get
            {
                object val = QueryMetadata("/app1/ifd/exif/subifd:{uint=36867}");
                if (val == null)
                {
                    return null;
                }
                else
                {
                    string date = (string)val;
                    try
                    {
                        return new DateTime(
                            int.Parse(date.Substring(0, 4)),    // year
                            int.Parse(date.Substring(5, 2)),    // month
                            int.Parse(date.Substring(8, 2)),    // day
                            int.Parse(date.Substring(11, 2)),   // hour
                            int.Parse(date.Substring(14, 2)),   // minute
                            int.Parse(date.Substring(17, 2))    // second
                        );
                    }
                    catch (FormatException)
                    {
                        return null;
                    }
                    catch (OverflowException)
                    {
                        return null;
                    }
                    catch (ArgumentNullException)
                    {
                        return null;
                    }
                    catch (NullReferenceException)
                    {
                        return null;
                    }
                }
            }

        }
    }

    public class VqaData
    {
        public string Description { get; }
        public string Images { get; private set; }
        public string Captions { get; private set; }
        public string PixelMaps { get; private set; }
        public string PythonHandler { get; private set; }

        public VqaData(string description, string images, string dataFile, string pixelMaps, string pythonHandler)
        {
            this.Description = description;
            this.Images = images;
            this.Captions = dataFile;
            this.PixelMaps = pixelMaps;
            this.PythonHandler = pythonHandler;
        }

        public override string ToString()
        {
            return $"{this.Description} ({Path.GetFileName(this.PythonHandler)} :   {Path.GetDirectoryName(this.Images)})";
        }

    }
}
