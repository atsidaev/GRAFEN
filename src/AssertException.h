#ifndef _ASSERT_EXCEPTION_H_
#define _ASSERT_EXCEPTION_H_

#include <string>
#include <exception>

#define Assert(exp) do { if (!(exp)) throw std::runtime_error("Assertion failed at: " + std::string(__FILE__) + " # line " + std::string(std::to_string(__LINE__))); } while (0)

#endif
