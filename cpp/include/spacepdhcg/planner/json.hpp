#pragma once

// Minimal strict JSON document model, parser, and writer for the planner.
//
// The planner exchanges one problem document and one result document per
// invocation, so the priority is correctness and unambiguous diagnostics rather
// than throughput.  Numbers are always stored as double; integers that fit in
// 53 bits round-trip exactly.  Non-finite doubles are written as `null`.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace spacepdhcg::planner::json {

class Value;

using Array = std::vector<Value>;
/// Insertion-ordered object so documents render deterministically.
using Object = std::vector<std::pair<std::string, Value>>;

class ParseError final : public std::runtime_error {
  public:
    explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

class TypeError final : public std::runtime_error {
  public:
    explicit TypeError(const std::string& message) : std::runtime_error(message) {}
};

enum class Kind : std::uint8_t { null, boolean, number, string, array, object };

class Value {
  public:
    Value() = default;
    Value(std::nullptr_t) : kind_(Kind::null) {}  // NOLINT(google-explicit-constructor)
    Value(bool value) : kind_(Kind::boolean), boolean_(value) {}  // NOLINT
    Value(double value) : kind_(Kind::number), number_(value) {}  // NOLINT
    Value(int value) : kind_(Kind::number), number_(static_cast<double>(value)) {}  // NOLINT
    Value(unsigned int value)  // NOLINT
        : kind_(Kind::number), number_(static_cast<double>(value)) {}
    Value(long value) : kind_(Kind::number), number_(static_cast<double>(value)) {}  // NOLINT
    Value(unsigned long value)  // NOLINT
        : kind_(Kind::number), number_(static_cast<double>(value)) {}
    Value(long long value)  // NOLINT
        : kind_(Kind::number), number_(static_cast<double>(value)) {}
    Value(unsigned long long value)  // NOLINT
        : kind_(Kind::number), number_(static_cast<double>(value)) {}
    Value(const char* value) : kind_(Kind::string), string_(value) {}  // NOLINT
    Value(std::string value) : kind_(Kind::string), string_(std::move(value)) {}  // NOLINT
    Value(std::string_view value) : kind_(Kind::string), string_(value) {}  // NOLINT
    Value(Array value)  // NOLINT
        : kind_(Kind::array), array_(std::make_shared<Array>(std::move(value))) {}
    Value(Object value)  // NOLINT
        : kind_(Kind::object), object_(std::make_shared<Object>(std::move(value))) {}

    [[nodiscard]] static Value array() { return Value(Array{}); }
    [[nodiscard]] static Value object() { return Value(Object{}); }

    template <typename Range>
    [[nodiscard]] static Value numbers(const Range& range) {
        Array result;
        for (const auto item : range) {
            result.emplace_back(static_cast<double>(item));
        }
        return Value(std::move(result));
    }

    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    [[nodiscard]] bool is_null() const noexcept { return kind_ == Kind::null; }
    [[nodiscard]] bool is_boolean() const noexcept { return kind_ == Kind::boolean; }
    [[nodiscard]] bool is_number() const noexcept { return kind_ == Kind::number; }
    [[nodiscard]] bool is_string() const noexcept { return kind_ == Kind::string; }
    [[nodiscard]] bool is_array() const noexcept { return kind_ == Kind::array; }
    [[nodiscard]] bool is_object() const noexcept { return kind_ == Kind::object; }

    [[nodiscard]] bool as_boolean() const {
        require(Kind::boolean, "boolean");
        return boolean_;
    }
    [[nodiscard]] double as_number() const {
        require(Kind::number, "number");
        return number_;
    }
    [[nodiscard]] const std::string& as_string() const {
        require(Kind::string, "string");
        return string_;
    }
    [[nodiscard]] const Array& as_array() const {
        require(Kind::array, "array");
        return *array_;
    }
    [[nodiscard]] Array& as_array() {
        require(Kind::array, "array");
        return *array_;
    }
    [[nodiscard]] const Object& as_object() const {
        require(Kind::object, "object");
        return *object_;
    }
    [[nodiscard]] Object& as_object() {
        require(Kind::object, "object");
        return *object_;
    }

    /// Object member lookup; returns nullptr when absent or when this is not an object.
    [[nodiscard]] const Value* find(std::string_view key) const noexcept {
        if (kind_ != Kind::object) {
            return nullptr;
        }
        for (const auto& [name, value] : *object_) {
            if (name == key) {
                return &value;
            }
        }
        return nullptr;
    }

    [[nodiscard]] bool contains(std::string_view key) const noexcept {
        return find(key) != nullptr;
    }

    [[nodiscard]] const Value& at(std::string_view key) const {
        const auto* value = find(key);
        if (value == nullptr) {
            throw TypeError("missing JSON member '" + std::string(key) + "'");
        }
        return *value;
    }

    /// Insert or replace a member, preserving first-insertion order.
    Value& set(std::string key, Value value) {
        if (kind_ == Kind::null) {
            kind_ = Kind::object;
            object_ = std::make_shared<Object>();
        }
        require(Kind::object, "object");
        for (auto& [name, existing] : *object_) {
            if (name == key) {
                existing = std::move(value);
                return *this;
            }
        }
        object_->emplace_back(std::move(key), std::move(value));
        return *this;
    }

    Value& push_back(Value value) {
        if (kind_ == Kind::null) {
            kind_ = Kind::array;
            array_ = std::make_shared<Array>();
        }
        require(Kind::array, "array");
        array_->push_back(std::move(value));
        return *this;
    }

    [[nodiscard]] std::size_t size() const {
        if (kind_ == Kind::array) {
            return array_->size();
        }
        if (kind_ == Kind::object) {
            return object_->size();
        }
        throw TypeError("JSON size() requires an array or object");
    }

  private:
    Kind kind_{Kind::null};
    bool boolean_{false};
    double number_{0.0};
    std::string string_{};
    std::shared_ptr<Array> array_{};
    std::shared_ptr<Object> object_{};

    void require(Kind expected, const char* label) const {
        if (kind_ != expected) {
            throw TypeError(
                std::string("JSON value is not a ") + label + " (actual kind: " + kind_name()
                + ")"
            );
        }
    }

    [[nodiscard]] const char* kind_name() const noexcept {
        switch (kind_) {
            case Kind::null:
                return "null";
            case Kind::boolean:
                return "boolean";
            case Kind::number:
                return "number";
            case Kind::string:
                return "string";
            case Kind::array:
                return "array";
            case Kind::object:
                return "object";
        }
        return "unknown";
    }
};

namespace detail {

class Parser {
  public:
    explicit Parser(std::string_view text) : text_(text) {}

    [[nodiscard]] Value parse_document() {
        skip_whitespace();
        auto value = parse_value(0U);
        skip_whitespace();
        if (position_ != text_.size()) {
            fail("trailing characters after JSON document");
        }
        return value;
    }

  private:
    static constexpr std::size_t maximum_depth = 256U;
    std::string_view text_;
    std::size_t position_{0U};

    [[noreturn]] void fail(const std::string& message) const {
        throw ParseError(
            "JSON parse error at byte " + std::to_string(position_) + ": " + message
        );
    }

    [[nodiscard]] bool at_end() const noexcept { return position_ >= text_.size(); }

    [[nodiscard]] char peek() const {
        if (at_end()) {
            fail("unexpected end of input");
        }
        return text_[position_];
    }

    char take() {
        const char value = peek();
        ++position_;
        return value;
    }

    void expect(char wanted) {
        const char actual = take();
        if (actual != wanted) {
            fail(std::string("expected '") + wanted + "' but found '" + actual + "'");
        }
    }

    void skip_whitespace() noexcept {
        while (!at_end()) {
            const char value = text_[position_];
            if (value == ' ' || value == '\t' || value == '\n' || value == '\r') {
                ++position_;
            } else {
                break;
            }
        }
    }

    [[nodiscard]] Value parse_value(std::size_t depth) {
        if (depth > maximum_depth) {
            fail("nesting depth exceeds the planner limit");
        }
        const char value = peek();
        switch (value) {
            case '{':
                return parse_object(depth + 1U);
            case '[':
                return parse_array(depth + 1U);
            case '"':
                return Value(parse_string());
            case 't':
                consume_literal("true");
                return Value(true);
            case 'f':
                consume_literal("false");
                return Value(false);
            case 'n':
                consume_literal("null");
                return Value(nullptr);
            default:
                break;
        }
        if (value == '-' || (value >= '0' && value <= '9')) {
            return Value(parse_number());
        }
        fail(std::string("unexpected character '") + value + "'");
    }

    void consume_literal(std::string_view literal) {
        if (text_.substr(position_, literal.size()) != literal) {
            fail("invalid literal; expected " + std::string(literal));
        }
        position_ += literal.size();
    }

    [[nodiscard]] Value parse_object(std::size_t depth) {
        expect('{');
        Object members;
        skip_whitespace();
        if (peek() == '}') {
            take();
            return Value(std::move(members));
        }
        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                fail("object keys must be strings");
            }
            auto key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            for (const auto& [existing, ignored] : members) {
                static_cast<void>(ignored);
                if (existing == key) {
                    fail("duplicate object key '" + key + "'");
                }
            }
            members.emplace_back(std::move(key), parse_value(depth));
            skip_whitespace();
            const char separator = take();
            if (separator == '}') {
                break;
            }
            if (separator != ',') {
                fail("expected ',' or '}' inside object");
            }
        }
        return Value(std::move(members));
    }

    [[nodiscard]] Value parse_array(std::size_t depth) {
        expect('[');
        Array items;
        skip_whitespace();
        if (peek() == ']') {
            take();
            return Value(std::move(items));
        }
        while (true) {
            skip_whitespace();
            items.push_back(parse_value(depth));
            skip_whitespace();
            const char separator = take();
            if (separator == ']') {
                break;
            }
            if (separator != ',') {
                fail("expected ',' or ']' inside array");
            }
        }
        return Value(std::move(items));
    }

    [[nodiscard]] unsigned int parse_hex4() {
        unsigned int result = 0U;
        for (int digit = 0; digit < 4; ++digit) {
            const char value = take();
            result <<= 4U;
            if (value >= '0' && value <= '9') {
                result |= static_cast<unsigned int>(value - '0');
            } else if (value >= 'a' && value <= 'f') {
                result |= static_cast<unsigned int>(value - 'a' + 10);
            } else if (value >= 'A' && value <= 'F') {
                result |= static_cast<unsigned int>(value - 'A' + 10);
            } else {
                fail("invalid \\u escape digit");
            }
        }
        return result;
    }

    static void append_utf8(std::string& output, unsigned int code_point) {
        if (code_point < 0x80U) {
            output.push_back(static_cast<char>(code_point));
        } else if (code_point < 0x800U) {
            output.push_back(static_cast<char>(0xC0U | (code_point >> 6U)));
            output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        } else if (code_point < 0x10000U) {
            output.push_back(static_cast<char>(0xE0U | (code_point >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        } else {
            output.push_back(static_cast<char>(0xF0U | (code_point >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        }
    }

    [[nodiscard]] std::string parse_string() {
        expect('"');
        std::string result;
        while (true) {
            const char value = take();
            if (value == '"') {
                return result;
            }
            if (static_cast<unsigned char>(value) < 0x20U) {
                fail("control character inside string");
            }
            if (value != '\\') {
                result.push_back(value);
                continue;
            }
            const char escape = take();
            switch (escape) {
                case '"':
                    result.push_back('"');
                    break;
                case '\\':
                    result.push_back('\\');
                    break;
                case '/':
                    result.push_back('/');
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                case 'u': {
                    unsigned int code_point = parse_hex4();
                    if (code_point >= 0xD800U && code_point <= 0xDBFFU) {
                        if (take() != '\\' || take() != 'u') {
                            fail("unpaired UTF-16 surrogate");
                        }
                        const unsigned int low = parse_hex4();
                        if (low < 0xDC00U || low > 0xDFFFU) {
                            fail("invalid UTF-16 low surrogate");
                        }
                        code_point = 0x10000U + ((code_point - 0xD800U) << 10U)
                            + (low - 0xDC00U);
                    } else if (code_point >= 0xDC00U && code_point <= 0xDFFFU) {
                        fail("unexpected UTF-16 low surrogate");
                    }
                    append_utf8(result, code_point);
                    break;
                }
                default:
                    fail("invalid escape sequence");
            }
        }
    }

    [[nodiscard]] double parse_number() {
        const std::size_t start = position_;
        if (peek() == '-') {
            take();
        }
        if (peek() == '0') {
            take();
        } else if (peek() >= '1' && peek() <= '9') {
            while (!at_end() && text_[position_] >= '0' && text_[position_] <= '9') {
                ++position_;
            }
        } else {
            fail("invalid number");
        }
        if (!at_end() && text_[position_] == '.') {
            ++position_;
            if (at_end() || text_[position_] < '0' || text_[position_] > '9') {
                fail("digits required after decimal point");
            }
            while (!at_end() && text_[position_] >= '0' && text_[position_] <= '9') {
                ++position_;
            }
        }
        if (!at_end() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (!at_end() && (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            if (at_end() || text_[position_] < '0' || text_[position_] > '9') {
                fail("digits required in exponent");
            }
            while (!at_end() && text_[position_] >= '0' && text_[position_] <= '9') {
                ++position_;
            }
        }
        const std::string token(text_.substr(start, position_ - start));
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0') {
            fail("number conversion failed for '" + token + "'");
        }
        if (!std::isfinite(value)) {
            fail("number '" + token + "' is out of the finite double range");
        }
        return value;
    }
};

inline void write_string(std::string& output, std::string_view text) {
    output.push_back('"');
    for (const char value : text) {
        switch (value) {
            case '"':
                output += "\\\"";
                break;
            case '\\':
                output += "\\\\";
                break;
            case '\b':
                output += "\\b";
                break;
            case '\f':
                output += "\\f";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(value) < 0x20U) {
                    char buffer[8];
                    std::snprintf(
                        buffer,
                        sizeof(buffer),
                        "\\u%04x",
                        static_cast<unsigned int>(static_cast<unsigned char>(value))
                    );
                    output += buffer;
                } else {
                    output.push_back(value);
                }
        }
    }
    output.push_back('"');
}

inline void write_number(std::string& output, double value) {
    if (!std::isfinite(value)) {
        output += "null";
        return;
    }
    if (value == std::floor(value) && std::fabs(value) < 9.007199254740992e15) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
        output += buffer;
        return;
    }
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    output += buffer;
}

inline void write_value(std::string& output, const Value& value, int indent, int level) {
    const auto newline = [&](int depth) {
        if (indent > 0) {
            output.push_back('\n');
            output.append(static_cast<std::size_t>(indent * depth), ' ');
        }
    };
    switch (value.kind()) {
        case Kind::null:
            output += "null";
            return;
        case Kind::boolean:
            output += value.as_boolean() ? "true" : "false";
            return;
        case Kind::number:
            write_number(output, value.as_number());
            return;
        case Kind::string:
            write_string(output, value.as_string());
            return;
        case Kind::array: {
            const auto& items = value.as_array();
            if (items.empty()) {
                output += "[]";
                return;
            }
            output.push_back('[');
            for (std::size_t index = 0U; index < items.size(); ++index) {
                if (index != 0U) {
                    output.push_back(',');
                }
                newline(level + 1);
                write_value(output, items[index], indent, level + 1);
            }
            newline(level);
            output.push_back(']');
            return;
        }
        case Kind::object: {
            const auto& members = value.as_object();
            if (members.empty()) {
                output += "{}";
                return;
            }
            output.push_back('{');
            for (std::size_t index = 0U; index < members.size(); ++index) {
                if (index != 0U) {
                    output.push_back(',');
                }
                newline(level + 1);
                write_string(output, members[index].first);
                output.push_back(':');
                if (indent > 0) {
                    output.push_back(' ');
                }
                write_value(output, members[index].second, indent, level + 1);
            }
            newline(level);
            output.push_back('}');
            return;
        }
    }
}

}  // namespace detail

[[nodiscard]] inline Value parse(std::string_view text) {
    detail::Parser parser(text);
    return parser.parse_document();
}

/// Serialise with `indent` spaces per level (0 = compact single line).
[[nodiscard]] inline std::string dump(const Value& value, int indent = 0) {
    std::string output;
    detail::write_value(output, value, indent, 0);
    return output;
}

// Typed member helpers -------------------------------------------------------

[[nodiscard]] inline double number_at(const Value& object, std::string_view key) {
    const auto& value = object.at(key);
    if (!value.is_number()) {
        throw TypeError("JSON member '" + std::string(key) + "' must be a number");
    }
    return value.as_number();
}

[[nodiscard]] inline double number_or(
    const Value& object,
    std::string_view key,
    double fallback
) {
    const auto* value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    if (!value->is_number()) {
        throw TypeError("JSON member '" + std::string(key) + "' must be a number");
    }
    return value->as_number();
}

[[nodiscard]] inline std::string string_at(const Value& object, std::string_view key) {
    const auto& value = object.at(key);
    if (!value.is_string()) {
        throw TypeError("JSON member '" + std::string(key) + "' must be a string");
    }
    return value.as_string();
}

[[nodiscard]] inline std::string string_or(
    const Value& object,
    std::string_view key,
    std::string fallback
) {
    const auto* value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    if (!value->is_string()) {
        throw TypeError("JSON member '" + std::string(key) + "' must be a string");
    }
    return value->as_string();
}

[[nodiscard]] inline bool boolean_or(const Value& object, std::string_view key, bool fallback) {
    const auto* value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    if (!value->is_boolean()) {
        throw TypeError("JSON member '" + std::string(key) + "' must be a boolean");
    }
    return value->as_boolean();
}

[[nodiscard]] inline std::vector<double> numbers_at(
    const Value& object,
    std::string_view key,
    std::size_t expected_size = 0U
) {
    const auto& value = object.at(key);
    if (!value.is_array()) {
        throw TypeError("JSON member '" + std::string(key) + "' must be an array of numbers");
    }
    std::vector<double> result;
    result.reserve(value.size());
    for (const auto& item : value.as_array()) {
        if (!item.is_number()) {
            throw TypeError(
                "JSON member '" + std::string(key) + "' must contain only numbers"
            );
        }
        result.push_back(item.as_number());
    }
    if (expected_size != 0U && result.size() != expected_size) {
        throw TypeError(
            "JSON member '" + std::string(key) + "' must contain exactly "
            + std::to_string(expected_size) + " numbers (found " + std::to_string(result.size())
            + ")"
        );
    }
    return result;
}

}  // namespace spacepdhcg::planner::json
