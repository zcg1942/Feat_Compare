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
    assert(!image.empty());//���ԣ����ʽ��ֵΪ�ٽ��˳������һ��������Ϣ

    if (featureEngine)//Ptr<cv::Feature2D>���ͣ����� FeatureDetector,DescriptorExtractor
		//����������push_back�˼����㷨��ΪfearureEngine
    {
        (*featureEngine)(image, cv::noArray(), kp, desc);//Ҫִ����һ�䣬���������Ϊ0
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
    matcher->match(query, train, matches);//�ͺ���������ȣ��������train��queryλ�û���
}

void FeatureAlgorithm::matchFeatures(const Descriptors& train, const Descriptors& query, int k, std::vector<Matches>& matches) const
{
    assert(knMatchSupported);
    matcher->knnMatch(query, train, matches, k);
}

