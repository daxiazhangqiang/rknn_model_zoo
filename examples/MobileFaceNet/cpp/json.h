#ifndef RKNN_MOBILEFACE_JSON_H_
#define RKNN_MOBILEFACE_JSON_H_

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

// 极简 JSON 解析 / 序列化工具，仅依赖 C++11 标准库。
// 只实现本项目需要的能力：object / array / string / number / bool / null。
namespace jsonmini {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object };

    Type type;
    bool b;
    double num;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    Value() : type(Null), b(false), num(0.0) {}

    bool is_null() const { return type == Null; }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }

    double as_number(double d = 0.0) const { return type == Number ? num : d; }
    int as_int(int d = 0) const { return type == Number ? static_cast<int>(num) : d; }
    const std::string& as_string() const {
        static std::string empty;
        return type == String ? str : empty;
    }

    const Value& get(const std::string& k) const {
        if (type == Object) {
            std::map<std::string, Value>::const_iterator it = obj.find(k);
            if (it != obj.end()) {
                return it->second;
            }
        }
        static Value null_value;
        return null_value;
    }

    const Value& at(size_t i) const {
        if (type == Array && i < arr.size()) {
            return arr[i];
        }
        static Value null_value;
        return null_value;
    }

    size_t size() const { return type == Array ? arr.size() : 0; }
};

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text), pos_(0) {}

    Value parse() {
        Value v;
        skip_ws();
        parse_value(v);
        return v;
    }

private:
    const std::string& text_;
    size_t pos_;

    void skip_ws() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos_++;
            } else {
                break;
            }
        }
    }

    char peek() {
        skip_ws();
        return pos_ < text_.size() ? text_[pos_] : '\0';
    }

    char take() {
        skip_ws();
        return pos_ < text_.size() ? text_[pos_++] : '\0';
    }

    void parse_value(Value& v) {
        char c = peek();
        if (c == '{') {
            parse_object(v);
        } else if (c == '[') {
            parse_array(v);
        } else if (c == '"') {
            v.type = Value::String;
            v.str = parse_string();
        } else if (c == 't') {
            for (int i = 0; i < 4; i++) take();
            v.type = Value::Bool;
            v.b = true;
        } else if (c == 'f') {
            for (int i = 0; i < 5; i++) take();
            v.type = Value::Bool;
            v.b = false;
        } else if (c == 'n') {
            for (int i = 0; i < 4; i++) take();
            v.type = Value::Null;
        } else {
            v.type = Value::Number;
            v.num = parse_number();
        }
    }

    void parse_object(Value& v) {
        v.type = Value::Object;
        take();  // {
        char c = peek();
        if (c == '}') {
            take();
            return;
        }
        while (true) {
            std::string key = parse_string();
            skip_ws();
            if (peek() == ':') {
                take();
            }
            Value child;
            parse_value(child);
            v.obj[key] = child;
            skip_ws();
            c = peek();
            if (c == ',') {
                take();
                continue;
            }
            if (c == '}') {
                take();
                break;
            }
            break;
        }
    }

    void parse_array(Value& v) {
        v.type = Value::Array;
        take();  // [
        char c = peek();
        if (c == ']') {
            take();
            return;
        }
        while (true) {
            Value child;
            parse_value(child);
            v.arr.push_back(child);
            skip_ws();
            c = peek();
            if (c == ',') {
                take();
                continue;
            }
            if (c == ']') {
                take();
                break;
            }
            break;
        }
    }

    std::string parse_string() {
        std::string out;
        take();  // "
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"') {
                break;
            }
            if (c == '\\' && pos_ < text_.size()) {
                char e = text_[pos_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        unsigned int cp = 0;
                        for (int i = 0; i < 4 && pos_ < text_.size(); i++) {
                            char h = text_[pos_++];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                        }
                        if (cp < 0x80) {
                            out += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    double parse_number() {
        size_t start = pos_;
        if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) {
            pos_++;
        }
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                pos_++;
            } else {
                break;
            }
        }
        return atof(text_.substr(start, pos_ - start).c_str());
    }
};

inline Value parse(const std::string& text) {
    Parser p(text);
    return p.parse();
}

inline std::string escape_string(const std::string& s) {
    std::string out;
    out += '"';
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default: out += c; break;
        }
    }
    out += '"';
    return out;
}

}  // namespace jsonmini

#endif  // RKNN_MOBILEFACE_JSON_H_
