#include <iostream>
#include <ORBextractor.h>
#include <ORBmatcher.h>
#include <Frame.h>
#include <opencv2/opencv.hpp>
#include <CameraModels/KannalaBrandt8.h>


using namespace std;
using namespace ORB_SLAM3;

int main() {
  //Read intrinsic parameters
  float fx = 705.96;
  float fy = 705.16;
  float cx = 649.48;
  float cy = 371.91;

  float k0 = -0.04295;
  float k1 = -0.023353;
  float k2 = -0.009984;
  float k3 = -0.0040688;
  vector<float> vCalibration = {fx,fy,cx,cy,k0,k1,k2,k3};
  GeometricCamera* mpCamera = new KannalaBrandt8(vCalibration);

  int nFeatures = 1000;
  float fScaleFactor = 1.2;
  int nLevels = 8;
  int fIniThFAST = 20;
  int fMinThFAST = 7;
  ORBextractor* mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
  ORBVocabulary* mpORBVocabulary = new ORBVocabulary();
  Frame mCurrentFrame;
  Frame mLastFrame;
  cv::Mat mDistCoef = cv::Mat::zeros(4,1,CV_32F);
  mDistCoef.at<float>(0) = 0;
  mDistCoef.at<float>(1) = 0;
  mDistCoef.at<float>(2) = 0;
  mDistCoef.at<float>(3) = 0;
  
  float mbf = 0.0;
  float mThDepth = 0.0;
  string img_path_1 = "/home/winston/data/rgb/1712487285_337013450.jpg";
  string img_path_2 = "/home/winston/data/rgb/1712487285_436432252.jpg";
  cv::Mat mImGrayLast = cv::imread(img_path_1, cv::IMREAD_GRAYSCALE);
  cv::Mat mImGrayCur = cv::imread(img_path_2, cv::IMREAD_GRAYSCALE);
  double timestamp = 0.0;
  mLastFrame = Frame(mImGrayLast,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
  mCurrentFrame = Frame(mImGrayCur,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
  // Find correspondences
  ORBmatcher matcher(0.9,true);
  std::vector<int> mvIniMatches;
  std::vector<cv::Point2f> mvbPrevMatched;
  mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
  for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
    mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;
  int nmatches = matcher.SearchForInitialization(mLastFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
  cout << "Number of matches: " << nmatches << endl;
  // Draw matches
  cv::Mat imMatches;
  // convert mvIniMatches to DMatch
  std::vector<cv::DMatch> mvIniMatchesDMatch;
  for(size_t i=0; i<mvIniMatches.size(); i++)
    if (mvIniMatches[i]>0)
      mvIniMatchesDMatch.push_back(cv::DMatch(i,mvIniMatches[i],0));
  cv::drawMatches(mImGrayLast,mLastFrame.mvKeys,mImGrayCur,mCurrentFrame.mvKeys,mvIniMatchesDMatch,imMatches);
  cv::imshow("Matches",imMatches);
  cv::waitKey(0);
  return 0;
}
