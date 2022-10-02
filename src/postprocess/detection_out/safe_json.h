#ifndef __ORION_JSON_H__
#define __ORION_JSON_H__

#include <json.hpp>
#include <iostream>
#include <fstream>
#include <type_traits>

namespace vision{

class InputJson : public nlohmann::json
{
public:
    InputJson(){};
    virtual ~InputJson(){};

    template<typename ValueType, typename std::enable_if<std::is_arithmetic<ValueType>::value, void>::type* = nullptr>
    static inline ValueType SafeGet(const nlohmann::json& src_json)
    {
        if(true == src_json.is_null())
            return static_cast<ValueType>(0);

        return src_json.get<ValueType>();
    };

    template<typename ValueType, typename std::enable_if<std::is_class<ValueType>::value, void>::type* = nullptr>
    static inline ValueType SafeGet(const nlohmann::json& src_json)
    {
        if(true == src_json.is_null())
        {
            ValueType val;
            return val;
        }

        return src_json.get<ValueType>();
    };

};//class InputJson

} //namespece vision

#endif