#include <boost/algorithm/string.hpp>

#include <filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <torch/script.h> // to load existing models

namespace fs = std::filesystem;

// Utility to list the files that match a pattern in a directory
std::vector<std::string> list_files(const std::string& dir,
                                    const std::string& pattern = ".*")
{
  std::regex regex_pattern(pattern);
  std::vector<std::string> result;
  for(auto& p: fs::recursive_directory_iterator(dir))
    {
    std::stringstream fname;
    fname << p;
    if(std::regex_search(fname.str(), regex_pattern))
      {
      auto clean_fname = fname.str();
      boost::algorithm::erase_all(clean_fname, "\"");
      result.emplace_back(clean_fname);
      }
    }
  return result;
}

// Splits a string using any of a list of separators
std::vector<std::string> string_split(const std::string& str,
                               const char* separators = "\t ")
{
  std::vector< std::string > tokens;
  boost::split(tokens, str, boost::is_any_of(separators),
               boost::token_compress_on);
  return tokens;
}
// Joins a vector of strings into a string
std::string string_join(const std::vector<std::string>& vos, char sep=' ')
{
  std::stringstream ss;
  ss << vos[0];
  for(size_t i=1; i<vos.size(); ++i)
    {
    ss << sep << vos[i];
    }
  return ss.str();
}

//Gets the list of the available jpeg images and makes the directories
//splitting train and valid and cats and dogs
std::vector<std::string>
get_available_files_and_make_dirs(const std::string& image_path)
{
  auto files = list_files(image_path, R"(.*\/*.jpg)");
  const auto no_of_images = files.size();
  std::cout << "Total no. of images = " << no_of_images << '\n';
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(files.begin(), files.end(), g);
  for(const auto& t : std::array<std::string,2>{"train", "valid"})
    for(const auto& folder : std::array<std::string,2>{"dog", "cat"})
    {
      const auto new_folder = image_path+"/"+t+"/"+folder;
      if(fs::exists(new_folder))
      {
        std::cout << new_folder << " exists; removing\n";
        fs::remove_all(new_folder);
      }
      fs::create_directories(new_folder);
    }
  return files;
}

//From a list fo files, gets 'count' samples starting from 'first' and puts them
// in the selected subfolder
void generate_samples(const std::vector<std::string> files, int first,
                      int count, std::string subfolder)
{
  for(auto i=first; i<first+count; ++i)
  {
    auto f = files[i];
    auto tokens = string_split(f, "/");
    auto image_path = string_join(std::vector<std::string>(tokens.begin(),
                                                           --tokens.end()),
                                  '/');
    auto f_name = tokens[tokens.size()-1];
    auto cat_or_dog = std::string(string_split(f_name,".")[0]);
    auto new_name = image_path+"/"+subfolder+"/"+cat_or_dog+"/"+f_name;
    fs::copy(f, new_name);
  }
}

// Reads an image using OpenCV
cv::Mat read_image(const std::string& im_fn)
{
  cv::Mat image;
  image = cv::imread(im_fn, CV_LOAD_IMAGE_COLOR);

  if(! image.data )
  {
    throw("Could not open or find the image\n");

  }
  return image;
}

//Displays an OpenCV image
void show_image(const cv::Mat& image)
{
  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Display window", image );
  cv::waitKey(0);
  return;
}

//Displays a image given its file name
void show_image_file(const std::string& im_fn)
{
  auto image = read_image(im_fn);
  show_image(image);
  return;
}

//Cleans the directory with the tranin and validation samples
void clean_samples(const std::string& image_path)
{
  fs::remove_all(image_path+"/train");
  fs::remove_all(image_path+"/valid");
}

//Gets the name of the class from the file name
std::string get_class_name(std::string fname)
{
  auto t = string_split(string_split(fname,".")[0],"/");
  return t[t.size()-1];
}

//Identity transform for an image
struct NoTransform {
  cv::Mat operator()(const cv::Mat& im)
  {
    return im;
  }
};

//Cats and Dogs transform which resizes to 224x224 and normalizes with hard
//values as in the book example
struct CDTransform {
  cv::Mat operator()(const cv::Mat& im)
  {
    cv::Mat tmp;
    cv::resize(im, tmp, cv::Size(224, 224), 0, 0, CV_INTER_LINEAR);
    cv::Mat planes[3];
    cv::split(tmp, planes);
    std::array<float,3> slopes{0.485, 0.456, 0.406};
    std::array<float,3> intercepts{0.229, 0.224, 0.225};
    for(size_t i{0}; i<3; ++i)
      planes[i] = planes[i]*slopes[i]-intercepts[i];
    cv::Mat outImg;
        cv::merge(planes, 3, outImg);
    return outImg;
  }
};



// This class gives acces to the image of a folder via a specific
// transform The images are read and transformed when they are asked
// for The given path is supposed to have one subfolder with the name
// of each class containing the images for each class
template <typename Transform = NoTransform>
class ImageFolder
{
public:
  ImageFolder(std::string path, Transform t) : folder_path{path},
                                             transform{t}
  {
    find_classes();
  }
  //Access an image and its label by index in the folder
  std::pair<cv::Mat, unsigned int> operator[](size_t idx)
  {
    const auto im_fn = files[idx];
    const auto label = get_label(im_fn);
    return {transform(read_image(im_fn)), label};
  }
  void shuffle()
  {
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(files.begin(), files.end(), g);
  }
  size_t size() const { return files.size(); }
  //Maps the class names to their label
  std::map<std::string, unsigned int> class_to_idx;
  //The list of available classes
  std::vector<std::string> classes;
  //The files containig the images
  std::vector<std::string> files;
protected:
  unsigned int get_label(const std::string& im_fn)
  {
    return class_to_idx[get_class_name(im_fn)];
  }

  void find_classes()
  {
    files = list_files(folder_path, R"(.*\/*.jpg)");
    std::transform(cbegin(files), cend(files),
                   std::back_inserter(classes),
                   get_class_name);
    auto last = std::unique(begin(classes), end(classes));
    classes.erase(last, end(classes));
    for(size_t i=0; i<classes.size(); ++i)
    {
      class_to_idx[classes[i]] = i;
    }

  }
  Transform transform;
  std::string folder_path;
};

//Syntactic sugar
template <typename T>
size_t len(T x) { return x.size(); };

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
  os << "[ ";
  auto before_last = --end(v);
  for(auto it=begin(v); it!=before_last; ++it)
    os << *it << ", ";
  os << *before_last << " ]";
  return os;
}

//Class providing batches of samples from a sample folder. Will have
//to implement some kind of batch forwarding to the network, I guess
template <typename SampleFolder>
class DataLoader
{
public:
  DataLoader(SampleFolder folder, bool shuffle = true, size_t batch_size = 64)
    : dataset{folder}, do_shuffle{shuffle}, batch_size{batch_size} {};

  SampleFolder& dataset;

private:
  bool do_shuffle = false;
  size_t batch_size = 64;
};


std::ostream& operator<<(std::ostream& os,
                         const std::map<std::string, unsigned int>& m)
{
  os << "{ ";
  auto before_last = --end(m);
  for(auto it=begin(m); it!=before_last; ++it)
    os << (*it).first << " : " << (*it).second <<", ";
  os << (*before_last).first << " : " << (*before_last).second <<" }";
  return os;
}


int main(int argc, char* argv[])
{
  const std::string image_path{"/home/inglada/stok/DATA/DogsAndCats/train"};
  auto files = get_available_files_and_make_dirs(image_path);
  const auto training_samples = 20;
  generate_samples(files, 0, training_samples, "train");
  const auto validation_samples = 20;
  generate_samples(files, training_samples,
                   training_samples+validation_samples, "valid");
  auto train = ImageFolder(image_path+"/train", CDTransform{});
  auto valid = ImageFolder(image_path+"/valid", CDTransform{});
  std::cout << train.class_to_idx << '\n';
  std::cout << train.classes << '\n';
  auto [image_sample, sample_label] = train[training_samples/2];
  std::cout << "The label of the image is " << sample_label << '\n';
  show_image(image_sample);
  auto train_data_gen = DataLoader{train, true, 64};
  auto valid_data_gen = DataLoader{valid, true, 64};

  // Deserialize the ScriptModule from a file using torch::jit::load().
  //https://pytorch.org/tutorials/advanced/cpp_export.html
  torch::jit::script::Module module = 
    torch::jit::load(argv[1]);
  //assert(module != nullptr);
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  auto output = module.forward(inputs).toTensor();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  clean_samples(image_path);
  return 0;

}
