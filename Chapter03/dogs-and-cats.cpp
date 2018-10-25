#include <boost/algorithm/string.hpp>

#include <filesystem>

#include <opencv2/core/core.hpp>
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

namespace fs = std::filesystem;

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

void show_image(const std::string& im_fn)
{
  cv::Mat image;
  image = cv::imread(im_fn, CV_LOAD_IMAGE_COLOR);

  if(! image.data )                
  {
    throw("Could not open or find the image\n");
    
  }
  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Display window", image ); 
  cv::waitKey(0);                                  
  return;
}

void clean_samples(const std::string& image_path)
{
  fs::remove_all(image_path+"/train");
  fs::remove_all(image_path+"/valid");
}

struct NoTransform {};
template <typename Transform = NoTransform>
class ImageFolder
{
public:
  ImageFolder(std::string path, Transform t) : folder_path{path},
                                             transform{t}
  {
    find_classes();
  }
  std::map<std::string, unsigned int> class_to_idx;
  std::vector<std::string> classes;
protected:
  void find_classes()
  {
    auto files = list_files(folder_path, R"(.*\/*.jpg)");
    std::transform(cbegin(files), cend(files), 
                   std::back_inserter(classes),
                   [](std::string fname)
                   {
                     auto t = string_split(string_split(fname,".")[0],"/");
                     return t[t.size()-1];
                   });
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


int main()
{
  const std::string image_path{"/home/inglada/stok/DATA/DogsAndCats/train"};
  auto files = get_available_files_and_make_dirs(image_path);
  const auto training_samples = 20;
  generate_samples(files, 0, training_samples, "train");
  const auto validation_samples = 20;
  generate_samples(files, training_samples, 
                   training_samples+validation_samples, "valid");

  //  show_image(files[0]);
  auto ImF = ImageFolder(image_path+"/train", NoTransform{});

  std::cout << ImF.class_to_idx << '\n';
  std::cout << ImF.classes << '\n';
  clean_samples(image_path);
  return 0;

}

