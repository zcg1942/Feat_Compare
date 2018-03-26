#include "FeatureAlgorithm.hpp"
#include <cassert>



static cv::Ptr<cv::flann::IndexParams> indexParamsForDescriptorType(int descriptorType, int defaultNorm)
{
    switch (defaultNorm)
    {
    case cv::NORM_L2:
        return cv::Ptr<cv::flann::IndexParams>(new cv::flann::KDTreeIndexParams());

    case cv::NORM_HAMMING:
        return cv::Ptr<cv::flann::IndexParams>(new cv::flann::LshIndexParams(20, 15, 2));

    default:
        CV_Assert(false && "Unsupported descriptor type");
    };
}

cv::Ptr<cv::DescriptorMatcher> matcherForDescriptorType(int descriptorType, int defaultNorm, bool bruteForce)
{
    if (bruteForce)
        return cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(defaultNorm, true));
    else
        return  cv::Ptr<cv::DescriptorMatcher>(new cv::FlannBasedMatcher(indexParamsForDescriptorType(descriptorType, defaultNorm)));
}

FeatureAlgorithm::FeatureAlgorithm(const std::string& n, cv::Ptr<cv::FeatureDetector> d, cv::Ptr<cv::DescriptorExtractor> e, bool useBruteForceMather)
: name(n)
, knMatchSupported(false)
, detector(d)
, extractor(e)
, matcher(matcherForDescriptorType(e->descriptorType(), cv::NORM_L2, useBruteForceMather))
{
    CV_Assert(d);
    CV_Assert(e);
}

FeatureAlgorithm::FeatureAlgorithm(const std::string& n, cv::Ptr<cv::Feature2D> fe, bool useBruteForceMather)
	: name(n)
	, knMatchSupported(false)
	, featureEngine(fe)
	, matcher(matcherForDescriptorType(fe->descriptorType(), cv::NORM_L2, useBruteForceMather))
{
   CV_Assert(fe);
}


bool FeatureAlgorithm::extractFeatures(const cv::Mat& image, Keypoints& kp, Descriptors& desc) const
{
    assert(!image.empty());//断言，表达式的值为假将退出并输出一条错误信息

    if (featureEngine)//Ptr<cv::Feature2D>类型，包含 FeatureDetector,DescriptorExtractor
		//在主函数中push_back了几种算法作为fearureEngine
    {
        (*featureEngine)(image, cv::noArray(), kp, desc);//要执行这一句，否则输出都为0
    }
    else
    {
        detector->detect(image, kp);
    
        if (kp.empty())
            return false;
    
        extractor->compute(image, kp, desc);
    }
    
    
    return kp.size() > 0;
}

void FeatureAlgorithm::matchFeatures(const Descriptors& train, const Descriptors& query, Matches& matches) const
{
    matcher->match(query, train, matches);//和函数参数相比，这里参数train和query位置换了
}

void FeatureAlgorithm::matchFeatures(const Descriptors& train, const Descriptors& query, int k, std::vector<Matches>& matches) const
{
    assert(knMatchSupported);
    matcher->knnMatch(query, train, matches, k);
}

