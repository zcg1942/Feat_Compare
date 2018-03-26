#include "CollectedStatistics.hpp"

#include <sstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <cassert>

template<typename T>
std::string quote(const T& t)
{
    std::ostringstream quoteStr;
    quoteStr << "\"" << t << "\"";
    return quoteStr.str();
}

std::ostream& tab(std::ostream& str)
{
    return str << "\t";
}

std::ostream& null(std::ostream& str)
{
    return str << "NULL";
}

FrameMatchingStatistics::FrameMatchingStatistics()//构造函数 初始化
{
    totalKeypoints = 0;
    argumentValue = 0;
    percentOfMatches = 0;
    ratioTestFalseLevel = 0;
    meanDistance = 0;
    stdDevDistance = 0;
    correctMatchesPercent = 0;
    recall = 0;
    precision = 0;
    consumedTimeMs = 0;
    homographyError = std::numeric_limits<float>::max();
    isValid = false;
}


bool FrameMatchingStatistics::tryGetValue(StatisticElement element, float& value) const
{
    if (!isValid)
        return false;
    
    switch (element)
    {
        case  StatisticsElementPointsCount:
            value = totalKeypoints;
            return true;
            
        case StatisticsElementPercentOfCorrectMatches:
            value = correctMatchesPercent * 100;
            return true;
            
        case StatisticsElementPercentOfMatches:
            value = percentOfMatches * 100;
            return true;
            
        case StatisticsElementMeanDistance:
            value = meanDistance;
            return true;
            
        case StatisticsElementHomographyError:
            value = homographyError;
            return true;
            
        case StatisticsElementMatchingRatio:
            value = matchingRatio();
            return true;
            
        case StatisticsElementPatternLocalization:
            value = patternLocalization();
            return true;

        case StatisticsElementPrecision:
            value = precision;
            return true;

        case StatisticsElementRecall:
            value = recall;
            return true;
        default:
            return false;
    }
}

std::ostream& FrameMatchingStatistics::writeElement(std::ostream& str, StatisticElement elem) const
{
    float value;
    
    if (tryGetValue(elem, value))
    {
        str << value << tab;
    }
    else
    {
        str << null << tab;
    }

    return str;
}

SingleRunStatistics& CollectedStatistics::getStatistics(std::string algorithmName, std::string transformationName)
{
    return m_allStats[std::make_pair(algorithmName, transformationName)];
}


CollectedStatistics::OuterGroup CollectedStatistics::groupByAlgorithmThenByTransformation() const
{
    OuterGroup result;

    for (std::map<Key, SingleRunStatistics>::const_iterator i = m_allStats.begin(); i != m_allStats.end(); ++i)
    {
        result[i->first.first][i->first.second] = &(i->second);
    }

    return result;
}

CollectedStatistics::OuterGroupLine CollectedStatistics::groupByTransformationThenByAlgorithm() const
{
    OuterGroup result;

    for (std::map<Key, SingleRunStatistics>::const_iterator i = m_allStats.begin(); i != m_allStats.end(); ++i)
    {
        result[i->first.second][i->first.first] = &(i->second);
    }

    OuterGroupLine line;

    for (OuterGroup::const_iterator tIter = result.begin(); tIter != result.end(); ++tIter)
    {
        std::string transformationName               = tIter->first;
        const CollectedStatistics::InnerGroup& inner = tIter->second;

        GroupedByArgument& lineStat = line[transformationName];

        std::vector<const SingleRunStatistics*> statitics;

        for (CollectedStatistics::InnerGroup::const_iterator algIter = inner.begin(); algIter != inner.end(); ++algIter)
        {
            std::string algName = algIter->first;

            lineStat.algorithms.push_back(algName);
            statitics.push_back(algIter->second);
        }

        const SingleRunStatistics& firstStat = *statitics.front();
        int argumentsCount = firstStat.size();

        for (int i=0; i < argumentsCount; i++)
        {
            Line l;
            l.argument = firstStat[i].argumentValue;

            for (int algIndex = 0; algIndex < statitics.size(); algIndex++)
            {
                const SingleRunStatistics& s = *statitics[algIndex];

                l.stats.push_back(&s[i]);
            }

            lineStat.lines.push_back(l);
        }
    }


    return line;
}

std::ostream& CollectedStatistics::printAverage(std::ostream& str, StatisticElement elem) const
{
    OuterGroup result;//OuterGroup是map，第二个参数是InnerGroup(也是map）
    str << "Average" <<elem<<std::endl;//根据不同elem输出不同变换下的值,elem是枚举类型,这样只会输出整型数
	switch (elem)
	{
	case 0:str << "  PointsCount" << std::endl; break;
	case 1:str << "  PercentOfCorrectMatches" << std::endl; break;
	case 2:str << " PercentOfMatches" << std::endl; break;
	case 3:str << "  MeanDistance" << std::endl; break;
	case 4:str << "MatchingRatio" << std::endl; break;
	case 5:str << " HomographyError" << std::endl; break;
	case 6:str << " PatternLocalization" << std::endl; break;
	case 7:str << "  AverageReprojectionError" << std::endl; break;
	case 8:str << " Recall" << std::endl; break;
	case 9:str << " Precision" << std::endl;

	}
    
    for (std::map<Key, SingleRunStatistics>::const_iterator i = m_allStats.begin(); i != m_allStats.end(); ++i)//key是pair，两个都是string，算法名字和变换名字
    {
        result[i->first.second][i->first.first] = &(i->second);
        
        str << i->first.first << tab << i->first.second << tab << average(i->second, elem) << std::endl;
		//i是map的迭代器，i->first是Key，Key又是pair，average第一个参数是要获得size
		//average函数里面有一个tryGetValue函数，根据elem不同得到不同值
    }
    
    return str;
}

std::ostream& CollectedStatistics::printStatistics(std::ostream& str, StatisticElement elem) const
{
    CollectedStatistics::OuterGroupLine report = groupByTransformationThenByAlgorithm();

    for (CollectedStatistics::OuterGroupLine::const_iterator tIter = report.begin(); tIter != report.end(); ++tIter)
    {
        std::string transformationName = tIter->first;//变换名字
        str << quote(transformationName) << std::endl;

        const GroupedByArgument& inner = tIter->second;//变换的变量参数
		//GroupedByArgument有两个参数，一个是algorithms，一个是lines

        str << "Argument" << tab;
        for (size_t i=0; i<inner.algorithms.size(); i++)
        {
            str << quote(inner.algorithms[i]) << tab;
        }
        str << std::endl;

        for (size_t i=0; i<inner.lines.size();i++)//lines又是一个结构体,包含 FrameMatchingStatistics*的stats
        {
            const Line& l = inner.lines[i];
            str << l.argument << tab;

            for (size_t j=0; j< l.stats.size(); j++)
            {
                const FrameMatchingStatistics& item = *l.stats[j];
                item.writeElement(str, elem);
            }

            str << std::endl;
        }
    }

    return str << std::endl;
}

std::ostream& CollectedStatistics::printPerformanceStatistics(std::ostream& str) const
{
    str << quote("Performance")               << std::endl;
    str << quote("Algorithm")                 << tab
        << quote("Average time per Frame")    << tab
        << quote("Average time per KeyPoint") << std::endl;

    CollectedStatistics::OuterGroup report = groupByAlgorithmThenByTransformation();

    for (CollectedStatistics::OuterGroup::const_iterator alg = report.begin(); alg != report.end(); ++alg)
    {
        std::vector<double> timePerFrames;
        std::vector<double> timePerKeyPoint;

        for (CollectedStatistics::InnerGroup::const_iterator tIter = alg->second.begin(); tIter != alg->second.end(); ++tIter)
        {
            const SingleRunStatistics& runStatistics = *tIter->second;//InnerGroup是map，second是 SingleRunStatistics
            for (size_t i=0; i<runStatistics.size(); i++)
            {
                if (runStatistics[i].isValid)
                {
                    timePerFrames.push_back(runStatistics[i].consumedTimeMs);//每帧图像消耗的时间
                    timePerKeyPoint.push_back(runStatistics[i].totalKeypoints > 0 ? (runStatistics[i].consumedTimeMs / runStatistics[i].totalKeypoints) : 0);//每个特征点消耗的时间
                }
            }
        }

        double avgPerFrame    = std::accumulate(timePerFrames.begin(),   timePerFrames.end(), 0.0)     / timePerFrames.size();
        double avgPerKeyPoint = std::accumulate(timePerKeyPoint.begin(), timePerKeyPoint.end(), 0.0) / timePerKeyPoint.size();

		str << quote(alg->first) << tab// << tab << tab//算法名字
            << avgPerFrame       << tab//<<tab<<tab<<tab//平均值
            << avgPerKeyPoint    << std::endl;
    }

    return str << std::endl;
}

float average(const SingleRunStatistics& statistics, StatisticElement element)
{
    std::vector<float> scores;
    
    for (size_t i = 0; i< statistics.size(); i++)
    {
        float value;
        bool valid = statistics[i].tryGetValue(element, value);
        
        if (valid)
        {
            scores.push_back(value);
        }
    }
    
    float sum     = std::accumulate(scores.begin(), scores.end(), 0.0f);
    float average = sum / scores.size();
    return average;
}

float maximum(const SingleRunStatistics& statistics, StatisticElement element)
{
    std::vector<float> scores;
    
    for (size_t i = 0; i< statistics.size(); i++)
    {
        float value;
        bool valid = statistics[i].tryGetValue(element, value);
        
        if (valid)
        {
            scores.push_back(value);
        }
    }
    
    assert(!scores.empty());
    
    float max = *std::max_element(scores.begin(), scores.end());
    return max;
}

