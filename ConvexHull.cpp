#include <iostream>
#include <cmath>
#include <iomanip>
#include <optional>
#include <deque>
#include <vector>
#include <memory>
#include <cstddef>
#include <set>
#include <algorithm>
#include <list>

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename NumberType>
NumberType Round(NumberType x) {
  const long double kAccuracy = 0.0'0'1;
  if (std::abs(x) < kAccuracy) {
    return 0;
  }
  return x;
}

template <typename NumberType>
struct Point;
template <typename NumberType>
struct Vector;
template <typename NumberType>
class Line;
template <typename NumberType>
class Ray;
template <typename NumberType>
class Segment;

template <typename NumberType,
    bool IsClockwise,
    template <typename, typename> typename ContainerType = std::deque,
    typename Alloc = std::allocator<Point<NumberType>>>
class Polygon;

template <typename NumberType>
struct Point {
  Point& operator+=(const Point& other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  Point& operator-=(const Point& other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }

  Point& operator*=(NumberType scalar) {
    x *= scalar;
    y *= scalar;
    return *this;
  }

  Point& operator/=(NumberType scalar) {
    x /= scalar;
    y /= scalar;
    return *this;
  }

  std::optional<Point> Intersection(const Point<NumberType>& point) const;
  std::optional<Point> Intersection(const Line<NumberType>& line) const;
  std::optional<Point> Intersection(const Ray<NumberType>& ray) const;
  std::optional<Point> Intersection(const Segment<NumberType>& segment) const;
  template <bool IsClockwise,
      template <typename, typename> typename ContainerType,
      typename Alloc>
  std::optional<Point> Intersection(const Polygon<NumberType,
                                                  IsClockwise,
                                                  ContainerType,
                                                  Alloc>& polygon) const;

  template <typename OtherNumberType>
  friend std::ostream& operator<<(std::ostream& os,
                                  const Point<OtherNumberType>& point);
  template <typename OtherNumberType>
  friend std::istream& operator>>(std::istream& is,
                                  Point<OtherNumberType>& point);

  NumberType x = 0;
  NumberType y = 0;
};

template <typename NumberType>
struct Vector {
  Vector(NumberType x, NumberType y) : x(x), y(y) {}
  Vector(const Point<NumberType>& begin, const Point<NumberType>& end) : x(
      end.x - begin.x),
                                                                         y(end.y
                                                                               - begin.y) {}
  Vector() = default;
  Vector(const Vector& vector) = default;
  Vector(Vector&& vector) noexcept = default;
  Vector& operator=(const Vector& vector) = default;
  Vector& operator=(Vector&& vector) noexcept = default;

  Vector& operator+=(const Vector& other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  Vector& operator-=(const Vector& other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }

  Vector& operator*=(NumberType X) {
    x *= X;
    y *= X;
    return *this;
  }

  Vector& operator/=(NumberType X) {
    x /= X;
    y /= X;
    return *this;
  }

  template <typename OtherNumberType>
  friend std::ostream& operator<<(std::ostream& os,
                                  const Vector<OtherNumberType>& vector);
  template <typename OtherNumberType>
  friend std::istream& operator>>(std::istream& is,
                                  Vector<OtherNumberType>& vector);

  NumberType x = 0;
  NumberType y = 0;
};

template <typename NumberType>
std::ostream& operator<<(std::ostream& os, const Point<NumberType>& point) {
  os << point.x << ' ' << point.y;
  return os;
}

template <typename NumberType>
std::istream& operator>>(std::istream& is, Point<NumberType>& point) {
  NumberType x = 0;
  NumberType y = 0;
  is >> x >> y;
  point = Point<NumberType>({x, y});
  return is;
}

template <typename NumberType>
Point<NumberType> operator+(const Point<NumberType>& first,
                            const Point<NumberType>& second) {
  auto copy = first;
  copy += second;
  return copy;
}

template <typename NumberType>
Point<NumberType> operator-(const Point<NumberType>& first,
                            const Point<NumberType>& second) {
  auto copy = first;
  copy -= second;
  return copy;
}

template <typename NumberType>
Point<NumberType> operator*(const Point<NumberType>& point, long double x) {
  auto copy = point;
  copy *= x;
  return copy;
}

template <typename NumberType>
Point<NumberType> operator/(const Point<NumberType>& point, long double x) {
  auto copy = point;
  copy /= x;
  return copy;
}

template <typename NumberType>
bool operator==(const Point<NumberType>& first,
                const Point<NumberType>& second) {
  return first.x == second.x && first.y == second.y;
}

template <typename NumberType>
bool operator!=(const Point<NumberType>& first,
                const Point<NumberType>& second) {
  return !(first == second);
}

template <typename NumberType>
std::ostream& operator<<(std::ostream& os, const Vector<NumberType>& vector) {
  os << vector.x << ' ' << vector.y;
  return os;
}

template <typename NumberType>
std::istream& operator>>(std::istream& is, Vector<NumberType>& vector) {
  NumberType x = 0;
  NumberType y = 0;
  is >> x >> y;
  vector = Vector<NumberType>({x, y});
  return is;
}

template <typename NumberType>
Vector<NumberType> operator+(const Vector<NumberType>& first,
                             const Vector<NumberType>& second) {
  auto copy = first;
  copy += second;
  return copy;
}

template <typename NumberType>
Vector<NumberType> operator-(const Vector<NumberType>& first,
                             const Vector<NumberType>& second) {
  auto copy = first;
  copy -= second;
  return copy;
}

template <typename NumberType>
Vector<NumberType> operator*(const Vector<NumberType>& vector, NumberType x) {
  auto copy = vector;
  copy *= x;
  return copy;
}

template <typename NumberType>
Vector<NumberType> operator/(const Vector<NumberType>& vector, NumberType x) {
  auto copy = vector;
  copy /= x;
  return copy;
}

bool operator==(const Vector<long double>& first,
                const Vector<long double>& second) {
  return Round(first.x - second.x) == 0 && Round(first.y - second.y) == 0;
}

bool operator==(const Vector<int>& first, const Vector<int>& second) {
  return first.x == second.x && first.y == second.y;
}

template <typename NumberType>
Point<NumberType> operator+(const Point<NumberType>& point,
                            const Vector<NumberType>& vector) {
  auto copy = point;
  copy.x += vector.x;
  copy.y += vector.y;
  return copy;
}

template <typename NumberType>
Point<NumberType> operator-(const Point<NumberType>& point,
                            const Vector<NumberType>& vector) {
  auto copy = point;
  copy.x -= vector.x;
  copy.y -= vector.y;
  return copy;
}

template <typename NumberType>
NumberType ABS(const Vector<NumberType>& vector) {
  auto x = vector.x;
  x *= x;
  auto y = vector.y;
  y *= y;
  auto abs = std::sqrt(x + y);
  return abs;
}

template <typename NumberType>
NumberType ScalarMul(const Vector<NumberType>& first,
                     const Vector<NumberType>& second) {
  auto scalar = first.x * second.x + first.y * second.y;
  return scalar;
}

template <typename NumberType>
NumberType VectorMul(const Vector<NumberType>& first,
                     const Vector<NumberType>& second) {
  auto scalar = first.x * second.y - first.y * second.x;
  return scalar;
}

template <typename NumberType>
NumberType VectorMul(const Point<NumberType>& first,
                     const Point<NumberType>& second) {
  auto scalar = first.x * second.y - first.y * second.x;
  return scalar;
}

template <typename NumberType>
class Line {
 public:
  Line(NumberType a, NumberType b, NumberType c)
      : a_(a), b_(b), c_(c), normal_(a_, b_), guiding_(-b_, a_) {}

  Line(const Point<NumberType>& first, const Point<NumberType>& second)
      : a_(first.y - second.y),
        b_(second.x - first.x),
        c_(VectorMul(first, second)),
        normal_(a_, b_),
        guiding_(-b_, a_) {}
  Line(const Point<NumberType>& point, const Vector<NumberType>& vector) : Line(
      point,
      point + vector) {}
  Line() = default;
  Line(const Line& line) = default;
  Line(Line&& line) noexcept = default;
  Line& operator=(const Line& line) = default;
  Line& operator=(Line&& line) noexcept = default;

  std::optional<Point<NumberType>> Intersection(const Point<NumberType>& point) const;
  std::optional<Point<NumberType>> Intersection(const Line<NumberType>& line) const;
  std::optional<Point<NumberType>> Intersection(const Ray<NumberType>& ray) const;
  std::optional<Point<NumberType>> Intersection(const Segment<NumberType>& segment) const;
  template <bool IsClockwise,
      template <typename, typename> typename ContainerType,
      typename Alloc>
  std::optional<Point<NumberType>> Intersection(const Polygon<NumberType,
                                                              IsClockwise,
                                                              ContainerType,
                                                              Alloc>& polygon) const;

  Vector<NumberType> GetNormal() const {
    return normal_;
  }

  Vector<NumberType> GetGuiding() const {
    return guiding_;
  }

  bool OnLine(const Point<NumberType>& point) const {
    return (Round(a_ * point.x + b_ * point.y + c_) == 0);
  }

  NumberType OrientedDistance(const Point<NumberType>& point) const {
    auto tmp = a_ * point.x + b_ * point.y + c_;
    tmp /= std::sqrt(a_ * a_ + b_ * b_);
    return tmp;
  }

  NumberType Distance(const Point<NumberType>& point) const {
    auto tmp = a_ * point.x + b_ * point.y + c_;
    tmp /= std::sqrt(a_ * a_ + b_ * b_);
    return std::abs(tmp);
  }

  NumberType Distance(const Line& other) const {
    auto det = VectorMul(normal_, other.normal_);
    if (Round(det) != 0) {
      return 0;
    }
    Point<NumberType> point;
    if (b_ == 0) {
      point.x = -c_ / a_;
    } else {
      point.y = -c_ / b_;
    }
    auto distance = other.Distance(point);
    return distance;
  }

  NumberType Substitute(const Point<NumberType>& point) const {
    auto tmp = a_ * point.x + b_ * point.y + c_;
    return tmp;
  }

 private:
  NumberType a_ = 0;
  NumberType b_ = 1;
  NumberType c_ = 0;
  Vector<NumberType> normal_;
  Vector<NumberType> guiding_;
};

template <typename NumberType>
class Ray {
 public:
  Ray(const Point<NumberType>& begin, const Vector<NumberType>& guiding)
      : begin_(begin),
        guiding_(guiding) {}
  Ray() = default;
  Ray(const Ray& ray) = default;
  Ray(Ray&& ray) noexcept = default;
  Ray& operator=(const Ray& ray) = default;
  Ray& operator=(Ray&& ray) noexcept = default;

  std::optional<Point<NumberType>> Intersection(const Point<NumberType>& point) {
    auto tmp_vector = Vector(point.x - begin_.x, point.y - begin_.y);
    if (Round(VectorMul(guiding_, tmp_vector)) == 0
        && ScalarMul(guiding_, tmp_vector) >= 0) {
      return point;
    }
    return std::nullopt;
  }

  std::optional<Point<NumberType>> Intersection(const Point<NumberType>& point) const;
  std::optional<Point<NumberType>> Intersection(const Line<NumberType>& line) const;
  std::optional<Point<NumberType>> Intersection(const Ray<NumberType>& ray) const;
  std::optional<Point<NumberType>> Intersection(const Segment<NumberType>& segment) const;
  template <bool IsClockwise,
      template <typename, typename> typename ContainerType,
      typename Alloc>
  std::optional<Point<NumberType>> Intersection(const Polygon<NumberType,
                                                              IsClockwise,
                                                              ContainerType,
                                                              Alloc>& polygon) const;

  std::optional<Point<NumberType>> Intersection(const Line<NumberType>& line) {
    auto intersection = line.Intersection(Line(begin_,
                                               Point({begin_.x + guiding_.x,
                                                      begin_.y + guiding_.y})));
    if (intersection) {
      return Intersection(intersection.value());
    }
    return std::nullopt;
  }

  Point<NumberType> GetBegin() const {
    return begin_;
  }

  Vector<NumberType> GetGuiding() const {
    return guiding_;
  }

 private:
  Point<NumberType> begin_;
  Vector<NumberType> guiding_;
};

template <typename NumberType>
class Segment {
 public:
  Segment(const Point<NumberType>& begin, const Point<NumberType>& end)
      : begin(begin),
        end(end) {}
  Segment() = default;
  Segment(const Segment& line) = default;
  Segment(Segment&& line) noexcept = default;
  Segment& operator=(const Segment& line) = default;
  Segment& operator=(Segment&& line) noexcept = default;

  std::optional<Point<NumberType>> Intersection(const Point<NumberType>& point) const;
  std::optional<Point<NumberType>> Intersection(const Line<NumberType>& line) const;
  std::optional<Point<NumberType>> Intersection(const Ray<NumberType>& ray) const;
  std::optional<Point<NumberType>> Intersection(const Segment<NumberType>& segment) const;
  template <bool IsClockwise,
      template <typename, typename> typename ContainerType,
      typename Alloc>
  std::optional<Point<NumberType>> Intersection(const Polygon<NumberType,
                                                              IsClockwise,
                                                              ContainerType,
                                                              Alloc>& polygon) const;

  Vector<NumberType> GetNormal() const {
    return {begin.x - end.x, begin.y - end.y};
  }

  enum SegmentIntersection {
    Infinite,
    Single,
    Empty,
  };

  SegmentIntersection CheckSegmentIntersection(const Segment& other) const {
    auto guiding1 = GetGuiding();
    auto guiding2 = GetGuiding();
    auto mul = VectorMul(guiding1, guiding2);
    auto intersection = Intersection(other);
    if (!intersection) {
      return Empty;
    }
    if (Round(mul) != 0) {
      return Infinite;
    }
    auto vector1 = Vector<NumberType>(begin, other.begin);
    auto vector2 = Vector<NumberType>(begin, other.end);
    auto scalar1 = ScalarMul(vector1, vector2);
    if (scalar1 < 0) {
      return Infinite;
    }
    auto scalar2 = ScalarMul(vector1, guiding1);
    auto scalar3 = ScalarMul(vector2, guiding1);
    if (scalar2 >= 0 && scalar3 >= 0) {
      return Single;
    }
    return Infinite;
  }

  Vector<NumberType> GetGuiding() const {
    return {end.x - begin.x, end.y - begin.y};
  }

  Point<NumberType> begin;
  Point<NumberType> end;
};

template <typename NumberType, bool IsClockwise,
    template <typename, typename> typename ContainerType,
    typename Alloc>
class Polygon {
 public:
  template <bool is_const>
  class Iterator;
  using iterator = Iterator<false>;                                // NOLINT
  using const_iterator = Iterator<true>;                           // NOLINT
  using reverse_iterator = std::reverse_iterator<iterator>;        // NOLINT
  using Container = ContainerType<Point<NumberType>,
                                  Alloc>;                   // NOLINT
  using container_allocator = typename Container::allocator_type;  // NOLINT
  using container_iterator = typename Container::iterator;  // NOLINT
  using container_const_iterator = typename Container::const_iterator;  // NOLINT
  using value_type = typename Container::value_type;               // NOLINT

  Polygon() = default;
  Polygon(const Polygon& polygon) = default;
  Polygon(Polygon&& polygon) noexcept = default;
  Polygon& operator=(const Polygon& polygon) = default;
  Polygon& operator=(Polygon&& polygon) noexcept = default;
  template <typename Iterator>
  Polygon(Iterator begin, Iterator end) : points_(begin, end) {}
  explicit Polygon(Container container) : points_(container) {}
  explicit Polygon(uint64_t points_number)
      : points_(points_number) {}

  uint64_t Size() const {
    return points_.size();
  }
  std::optional<Point<NumberType>> Intersection(const Point<NumberType>& point) const;
  std::optional<Point<NumberType>> Intersection(const Line<NumberType>& line) const;
  std::optional<Point<NumberType>> Intersection(const Ray<NumberType>& ray) const;
  std::optional<Point<NumberType>> Intersection(const Segment<NumberType>& segment) const;
  template <bool OtherIsClockwise,
      template <typename, typename> typename OtherContainerType,
      typename OtherAlloc>
  std::optional<Point<NumberType>> Intersection(const Polygon<NumberType,
                                                              IsClockwise,
                                                              ContainerType,
                                                              Alloc>& polygon) const;

  bool IsConvex() const {
    if (precalculated_) {
      return is_convex_;
    }
    if (points_.size() < 4) {
      return is_convex_ = true;
    }
    bool rotation = true;
    uint64_t start = 0;
    for (; start < points_.size(); ++start) {
      auto point1 = points_[start % points_.size()];
      auto point2 = points_[(start + 1) % points_.size()];
      auto point3 = points_[(start + 2) % points_.size()];
      auto vector1 = Vector(point2.x - point1.x, point2.y - point1.y);
      auto vector2 = Vector(point3.x - point2.x, point3.y - point2.y);
      auto mul = VectorMul(vector1, vector2);
      if (mul != 0) {
        rotation = mul > 0;
        break;
      }
    }
    for (uint64_t pos = 0; pos < points_.size(); ++pos) {
      auto point1 = points_[(start + pos) % points_.size()];
      auto point2 = points_[(start + pos + 1) % points_.size()];
      auto point3 = points_[(start + pos + 2) % points_.size()];
      auto vector1 = Vector(point2.x - point1.x, point2.y - point1.y);
      auto vector2 = Vector(point3.x - point2.x, point3.y - point2.y);
      auto mul = VectorMul(vector1, vector2);
      if (rotation && mul >= 0 || !rotation && mul <= 0) {
        continue;
      }
      return is_convex_ = false;
    }
    return is_convex_ = true;
  }

  void Homothety(NumberType scalar) {
    for (auto& point: points_) {
      point /= scalar;
    }
  }

  void Compress(NumberType scalar) {
    for (auto& point: points_) {
      point /= scalar;
    }
  }

  uint64_t NumberOfTriangulations(const uint64_t mod) const {
    std::vector<std::vector<uint64_t>>
        dp(points_.size(), std::vector<uint64_t>(points_.size(), 0));
    auto are_visible = Diagonals();
    for (uint64_t left = 0; left + 1 < dp.size(); ++left) {
      dp[left][left + 1] = 1;
    }
    for (uint64_t len = 2; len < points_.size(); ++len) {
      for (uint64_t left = 0; left + len < points_.size(); ++left) {
        uint64_t right = left + len;
        if (are_visible[left][right]) {
          for (uint64_t j = left + 1; j < right; ++j) {
            auto tmp = dp[left][j] * dp[j][right];
            tmp %= mod;
            dp[left][right] += tmp;
            dp[left][right] %= mod;
          }
        } else {
          dp[left][right] = 0;
        }
      }
    }
    return dp[0][points_.size() - 1];
  }

  std::vector<std::vector<bool>> Diagonals() const {
    if (!precalculated_) {
      IsConvex();
      IsClockWise();
      precalculated_ = true;
    }
    std::vector<std::vector<bool>>
        answer(points_.size(), std::vector<bool>(points_.size(), false));
    for (uint64_t begin = 0; begin < points_.size(); ++begin) {
      answer[begin][((begin + 1 + points_.size()) % points_.size()) % points_.size()] = true;
      answer[((begin + 1) % points_.size() + points_.size()) % points_.size()][begin] = true;
      answer[(begin + points_.size() - 1) % points_.size()][begin] = true;
      answer[begin][(begin + points_.size() - 1) % points_.size()] = true;
      for (uint64_t end = begin + 2; end < points_.size(); ++end) {
        auto diagonal = Segment((*this)[begin], (*this)[end]);
        auto diagonal_guiding = diagonal.GetGuiding();
        bool has_intersection = false;
        for (uint64_t pos = 0; pos < points_.size(); ++pos) {
          auto point1 = (*this)[pos];
          auto point2 = (*this)[pos + 1];
          auto side = Segment<NumberType>(point1, point2);
          auto side_guiding = side.GetGuiding();
          if (pos != begin && pos != end && diagonal.Intersection(point1) ||
              (pos + points_.size() + 1) % points_.size() != begin && (pos + points_.size() + 1) % points_.size() != end && diagonal.Intersection(point2)) {
            has_intersection = true;
            break;
          }
          if (pos == begin || pos == end || (pos + 1) % points_.size() == begin || (pos + 1) % points_.size() == end) {
            continue;
          }
          auto intersection = side.Intersection(diagonal);
          if (intersection) {
            has_intersection = true;
            break;
          }
        }
        if (has_intersection) {
          continue;
        }
        answer[begin][end] = CheckDiagonal(begin, end);
        answer[end][begin] = answer[begin][end];
      }
    }
    return std::move(answer);
  }

  bool CheckDiagonal(size_t begin, size_t end) const {
    auto cur = (*this)[begin];
    auto next = (*this)[begin + 1];
    auto prev = (*this)[begin - 1];
    auto point_end = (*this)[end];
    auto vector1 = Vector<NumberType>(cur, prev);
    auto vector2 = Vector<NumberType>(cur, next);
    auto diagonal_guiding = Vector<NumberType>(cur, point_end);
    auto mul1 = VectorMul(vector1, vector2);
    auto mul3 = VectorMul(vector2, diagonal_guiding);
    bool tmp = PointBetweenTwoRelative(cur, prev, next, point_end);
    if (Round(mul1) == 0) {
      return is_clockwise_ && mul3 <= 0 || !is_clockwise_ && mul3 > 0;
    }
    bool is_valid = false;
    if (Round(mul1) > 0 && tmp || mul1 < 0 && !tmp) {
      is_valid = true;
    }
    if (!is_clockwise_) {
      is_valid = !is_valid;
    }
    return is_valid;
  }

  bool IsClockWise() const {
    auto left_most = LeftMost();
    auto prev = (*this)[left_most - 1];
    auto cur = (*this)[left_most];
    auto next = (*this)[left_most + 1];
    return is_clockwise_ = ClockwiseRotation(cur, prev, next);
//    NumberType sum = 0;
//    for (uint64_t pos = 0; pos < points_.size(); ++pos) {
//      auto point1 = (*this)[pos];
//      auto point2 = (*this)[pos + 1];
//      sum += (point2.x - point1.x) * (point2.y + point1.y);
//    }
//    return is_clockwise_ = sum >= 0;
  }

  size_t LeftMost() const {
    auto answer_num = 0;
    auto answer = points_[answer_num];
    for (size_t pos = 1; pos < points_.size(); ++pos) {
      auto point = points_[pos];
      if (point.x < answer.x) {
        answer_num = pos;
        answer = point;
        continue;
      }
      if (point.x == answer.x && point.y < answer.y) {
        answer_num = pos;
        answer = point;
      }
    }
    return answer_num;
  }

  enum PointIntersection {
    Inside,
    Outside,
    Boundary
  };

  PointIntersection CheckPoint(const Point<NumberType>& point) const {
    if (!precalculated_) {
      IsConvex();
      precalculated_ = true;
    }
    if (is_convex_) {
      return CheckPointInConvex(point);
    }
    Ray<NumberType> ray = Ray(point, {0, 1});
    uint64_t intersection_counter = 0;
    for (uint64_t pos = 0; pos < points_.size(); ++pos) {
      Point<NumberType> begin = (*this)[pos];
      Point<NumberType> end = (*this)[pos + 1];
      Segment side = Segment(begin, end);
      auto on_side = side.Intersection(point);
      if (on_side) {
        return PointIntersection::Boundary;
      }
      if (begin.y > end.y) {
        std::swap(begin, end);
      }
      if (begin.y > point.y || end.y <= point.y) {
        continue;
      }
      auto mul = Round(VectorMul(point - begin, end - begin));
      if (mul <= 0) {
        ++intersection_counter;
      }
    }
    if (intersection_counter % 2 == 1) {
      return PointIntersection::Inside;
    }
    return PointIntersection::Outside;
  }

  void Insert(const_iterator iterator, const Point<NumberType>& point) {
    precalculated_ = false;
    points_.insert(iterator.iterator_, point);
  }

  void Erase(const_iterator iterator) {
    precalculated_ = false;
    points_.erase(iterator.iterator_);
  }

  void PopBack() {
    Erase(--points_.end());
  }

  void PopFront() {
    Erase(points_.begin());
  }

  void PushBack(const Point<NumberType>& point) {
    Insert(points_.end(), point);
  }

  void PushFront(const Point<NumberType>& point) {
    Insert(points_.begin(), point);
  }

  void AddPoint(const Point<NumberType>& point) {
    points_.push_back(point);
  }

  Point<NumberType>& operator[](int pos) {
    precalculated_ = false;
    pos %= (int) points_.size();
    if (pos >= 0) {
      return points_[pos];
    }
    return points_[points_.size() + pos];
  }

  const Point<NumberType>& operator[](int pos) const {
    pos %= (int) points_.size();
    if (pos >= 0) {
      return points_[pos];
    }
    return points_[points_.size() + pos];
  }

  iterator begin() { return iterator(points_.begin()); }  // NOLINT

  const_iterator begin() const {  // NOLINT
    return (const_cast<Polygon<NumberType,
                               IsClockwise,
                               ContainerType,
                               Alloc>*>(this))->begin();
  }

  reverse_iterator rbegin() { return reverse_iterator(begin()); }  // NOLINT

  iterator end() { return iterator(points_.end()); }  // NOLINT

  const_iterator end() const {  // NOLINT
    return (const_cast<Polygon<NumberType,
                               IsClockwise,
                               ContainerType,
                               Alloc>*>(this))->end();
  }
  reverse_iterator rend() { return reverse_iterator(end()); }  // NOLINT
 private:
  PointIntersection CheckPointInConvex(const Point<NumberType>& point) const {
    auto origin = points_[0];
    int64_t left = 1;
    int64_t right = points_.size() - 1;
    if (Segment<NumberType>(origin, points_[right]).Intersection(point) ||
        Segment<NumberType>(origin, points_[left]).Intersection(point)) {
      return PointIntersection::Boundary;
    }
    if (!PointBetweenTwoRelative(points_[0],
                                 points_[left],
                                 points_[right],
                                 point)) {
      return PointIntersection::Outside;
    }
    while (right > left + 1) {
      int64_t mid = (right + left) / 2;
      auto orientation = ClockwiseRotation(origin, points_[mid], point);
      if (!is_clockwise_ && orientation || is_clockwise_ && !orientation) {
        right = mid;
      } else {
        left = mid;
      }
    }
    right = left + 1;
    if (Segment<NumberType>(points_[left],
                            points_[right]).Intersection(point)) {
      return PointIntersection::Boundary;
    }
    if (PointInTriangle(points_[0], points_[left], points_[right], point)) {
      return PointIntersection::Inside;
    }
    return PointIntersection::Outside;
  }

  bool PointInTriangle(const Point<NumberType>& point1,
                       const Point<NumberType>& point2,
                       const Point<NumberType>& point3,
                       const Point<NumberType>& point) const {
    auto vector1 = Vector<NumberType>(point1, point2);
    auto vector2 = Vector<NumberType>(point1, point3);
    auto vector3 = Vector<NumberType>(point1, point);
    auto s1 = std::abs(VectorMul(vector1, vector2));
    vector1 = Vector<NumberType>(point, point1);
    vector2 = Vector<NumberType>(point, point2);
    vector3 = Vector<NumberType>(point, point3);
    s1 -= std::abs(VectorMul(vector1, vector2));
    s1 -= std::abs(VectorMul(vector2, vector3));
    s1 -= std::abs(VectorMul(vector3, vector1));
    return Round(s1) == 0;
  }

  bool PointBetweenTwoRelative(const Point<NumberType>& origin,
                               const Point<NumberType>& from,
                               const Point<NumberType>& to,
                               const Point<NumberType>& between) const {
    auto vector_from = Vector<NumberType>(origin, from);
    auto vector_to = Vector<NumberType>(origin, to);
    auto vector_between = Vector<NumberType>(origin, between);
    bool left = VectorMul(vector_from, vector_between)
        * VectorMul(vector_from, vector_to) >= 0;
    bool right =
        VectorMul(vector_to, vector_between) * VectorMul(vector_to, vector_from)
            >= 0;
    return (left && right);
  }

  bool ClockwiseRotation(const Point<NumberType>& origin,
                         const Point<NumberType>& from,
                         const Point<NumberType>& to) const {
    auto vector1 = Vector<NumberType>(origin, from);
    auto vector2 = Vector<NumberType>(origin, to);
    auto mul = Round(VectorMul(vector1, vector2));
    return mul >= 0;
  }

  bool mutable precalculated_ = false;
  bool mutable is_clockwise_ = false;
  bool mutable is_convex_ = false;

  Container points_;
};

template <typename NumberType, bool IsClockwise,
    template <typename, typename> typename ContainerType,
    typename Alloc>
Polygon<NumberType, IsClockwise, ContainerType, Alloc> operator+(
    const Polygon<NumberType, IsClockwise, ContainerType, Alloc>& polygon1,
    const Polygon<NumberType, IsClockwise, ContainerType, Alloc>& polygon2) {
  if (!polygon1.IsConvex() || !polygon2.IsConvex()) {
    throw (std::invalid_argument("Non convex polygon"));
  }
  auto start1 = polygon1.LeftMost();
  auto start2 = polygon2.LeftMost();
  auto end1 = start1 + polygon1.Size();
  auto end2 = start2 + polygon2.Size();
  Polygon<NumberType, IsClockwise, ContainerType, Alloc> sum_polygon;
  auto last_added_point = polygon1[start1] + polygon2[start2];
  sum_polygon.PushBack(last_added_point);
  while (start1 < end1 || start2 < end2) {
    auto vector1 = Vector<NumberType>(polygon1[start1], polygon1[start1 + 1]);
    auto vector2 = Vector<NumberType>(polygon2[start2], polygon2[start2 + 1]);
    auto vector_mul = VectorMul(vector1, vector2);
    if (vector_mul > 0) {
      ++start1;
      last_added_point = last_added_point + vector1;
    } else {
      ++start2;
      last_added_point = last_added_point + vector2;
    }
    uint64_t sum_polygon_size = sum_polygon.Size();
    vector1 = Vector<NumberType>(sum_polygon[sum_polygon_size - 2],
                                 sum_polygon[sum_polygon_size - 1]);
    vector2 =
        Vector<NumberType>(sum_polygon[sum_polygon_size - 1], last_added_point);
    if (sum_polygon_size > 1) {
      if (Round(VectorMul(vector1, vector2)) == 0) {
        sum_polygon.PopBack();
      }
    }
    sum_polygon.PushBack(last_added_point);
  }

  sum_polygon.PopBack();
  return std::move(sum_polygon);
}

template <typename NumberType>
std::optional<Point<NumberType>> Point<NumberType>::Intersection(const Point<
    NumberType>& point) const {
  if ((*this) == point) {
    return *this;
  }
  return std::nullopt;
}

template <typename NumberType>
std::optional<Point<NumberType>> Point<NumberType>::Intersection(const Line<
    NumberType>& line) const {
  return line.Intersection(*this);
}

template <typename NumberType>
std::optional<Point<NumberType>> Point<NumberType>::Intersection(const Ray<
    NumberType>& ray) const {
  return ray.Intersection(*this);
}
template <typename NumberType>
std::optional<Point<NumberType>> Point<NumberType>::Intersection(const Segment<
    NumberType>& segment) const {
  return segment.Intersection(*this);
}

template <typename NumberType>
template <bool IsClockwise, template <typename, typename> typename ContainerType, typename Alloc>
std::optional<Point<NumberType>> Point<NumberType>::Intersection(const Polygon<
    NumberType,
    IsClockwise,
    ContainerType,
    Alloc>& polygon) const {
  return polygon.Intersection(*this);
}

template <typename NumberType>
std::optional<Point<NumberType>> Line<NumberType>::Intersection(const Point<
    NumberType>& point) const {
  if (Round(Substitute(point)) == 0) {
    return point;
  }
  return std::nullopt;
}

template <typename NumberType>
std::optional<Point<NumberType>> Line<NumberType>::Intersection(const Line<
    NumberType>& line) const {
  auto det = VectorMul(guiding_, line.guiding_);
  det = Round(det);
  if (det == 0) {
    if (Round(a_) == 0) {
      Point<NumberType> point({0, -c_ / b_});
      if (line.OnLine(point)) {
        return point;
      }
      return std::nullopt;
    }
    Point<NumberType> point({-c_ / a_, 0});
    if (line.OnLine(point)) {
      return point;
    }
    return std::nullopt;
  }
  Vector vector_a(a_, line.a_);
  Vector vector_b(b_, line.b_);
  Vector vector_c(-c_, -line.c_);
  auto det1 = VectorMul(vector_c, vector_b);
  auto det2 = VectorMul(vector_a, vector_c);
  auto x = det1 / det;
  auto y = det2 / det;
  Point<NumberType> point({x, y});
  return point;
}
template <typename NumberType>
std::optional<Point<NumberType>> Line<NumberType>::Intersection(const Ray<
    NumberType>& ray) const {
  return ray.Intersection(*this);
}
template <typename NumberType>
std::optional<Point<NumberType>> Line<NumberType>::Intersection(const Segment<
    NumberType>& segment) const {
  return segment.Intersection(*this);
}

template <typename NumberType>
template <bool IsClockwise, template <typename, typename> typename ContainerType, typename Alloc>
std::optional<Point<NumberType>> Line<NumberType>::Intersection(const Polygon<
    NumberType,
    IsClockwise,
    ContainerType,
    Alloc>& polygon) const {
  return std::optional<Point<NumberType>>();
}

template <typename NumberType>
std::optional<Point<NumberType>> Ray<NumberType>::Intersection(const Point<
    NumberType>& point) const {
  std::optional<Point<NumberType>>
      intersection = Line(begin_, guiding_).Intersection(point);
  if (intersection) {
    auto scalar_mul = ScalarMul(guiding_, Vector(begin_, intersection.value()));
    if (scalar_mul >= 0) {
      return intersection;
    }
  }
  return std::nullopt;
}

template <typename NumberType>
std::optional<Point<NumberType>> Ray<NumberType>::Intersection(const Segment<
    NumberType>& segment) const {
  return segment.Intersection(*this);
}

template <typename NumberType>
std::optional<Point<NumberType>> Ray<NumberType>::Intersection(const Line<
    NumberType>& line) const {
  std::optional<Point<NumberType>>
      intersection = line.Intersection(Line(begin_, guiding_));
  if (Round(VectorMul(line.GetGuiding(), guiding_)) == 0 &&
      intersection) {
    return begin_;
  }
  if (intersection) {
    auto scalar_mul = ScalarMul(guiding_, Vector(begin_, intersection.value()));
    if (scalar_mul >= 0) {
      return intersection;
    }
  }
  return std::nullopt;
}

template <typename NumberType>
std::optional<Point<NumberType>> Ray<NumberType>::Intersection(const Ray<
    NumberType>& ray) const {
  auto ray_line1 = Line(begin_, guiding_);
  auto ray_line2 = Line(ray.begin_, ray.guiding_);
  auto intersection = ray_line1.Intersection(ray_line2);
  if (Round(VectorMul(guiding_, ray.guiding_)) == 0 && intersection) {
    if (ScalarMul(guiding_, Vector(begin_, ray.begin_)) >= 0) {
      return ray.begin_;
    }
    return begin_;
  }
  if (intersection) {
    if (ScalarMul(guiding_, Vector(begin_, intersection.value())) >= 0 &&
        ScalarMul(ray.guiding_, Vector(ray.begin_, intersection.value()))
            >= 0) {
      return intersection;
    }
  }
  return std::nullopt;
}

template <typename NumberType>
template <bool IsClockwise, template <typename, typename> typename ContainerType, typename Alloc>
std::optional<Point<NumberType>> Ray<NumberType>::Intersection(const Polygon<
    NumberType,
    IsClockwise,
    ContainerType,
    Alloc>& polygon) const {
  throw std::invalid_argument("Is not implemented");
  return std::optional<Point<NumberType>>();
}

template <typename NumberType>
std::optional<Point<NumberType>> Segment<NumberType>::Intersection(const Point<
    NumberType>& point) const {
  auto guiding = GetGuiding();
  auto tmp_vector1 = Vector(begin, point);
  auto tmp_vector2 = Vector(point, end);
  if (Round(VectorMul(guiding, tmp_vector1)) == 0 &&
      ScalarMul(guiding, tmp_vector1) >= 0 &&
      ScalarMul(guiding, tmp_vector2) >= 0) {
    return point;
  }
  return std::nullopt;
}
template <typename NumberType>
std::optional<Point<NumberType>> Segment<NumberType>::Intersection(const Line<
    NumberType>& line) const {
  auto guiding = GetGuiding();
  auto segment_line = Line(begin, end);
  auto intersection = segment_line.Intersection(line);
  if (Round(VectorMul(guiding, line.GetGuiding())) == 0 && intersection) {
    return begin;
  }
  if (!intersection) {
    return std::nullopt;
  }
  auto side = line.Substitute(begin) * line.Substitute(end);
  if (side > 0) {
    return std::nullopt;
  }
  return intersection.value();
}
template <typename NumberType>
std::optional<Point<NumberType>> Segment<NumberType>::Intersection(const Ray<
    NumberType>& ray) const {
  auto ray_line = Line(ray.GetBegin(), ray.GetGuiding());
  auto intersection = Intersection(ray_line);
  if (intersection && Intersection(intersection.value())) {
    return intersection;
  }
  return std::nullopt;
}
template <typename NumberType>
std::optional<Point<NumberType>> Segment<NumberType>::Intersection(const Segment<
    NumberType>& segment) const {
  auto guiding = GetGuiding();
  auto line1 = Line(begin, end);
  auto line2 = Line(segment.begin, segment.end);
  if (Round(VectorMul(guiding, segment.GetGuiding())) == 0) {
    auto vector1 = Vector(begin, segment.begin);
    if (Round(VectorMul(guiding, vector1)) != 0) {
      return std::nullopt;
    }
    if (ScalarMul(guiding, vector1) >= 0
        && ScalarMul(guiding, guiding - vector1) >= 0) {
      return segment.begin;
    }
    auto vector2 = Vector(begin, segment.end);
    if (ScalarMul(guiding, vector2) >= 0
        && ScalarMul(guiding, guiding - vector2) >= 0) {
      return segment.end;
    }
    if (ScalarMul(vector1, vector2) <= 0) {
      return begin;
    }
    return std::nullopt;
  }
  auto intersection = line1.Intersection(line2);
  if (!intersection) {

    return std::nullopt;
  }
  auto left = line2.Substitute(begin) * line2.Substitute(end);
  auto right = line1.Substitute(segment.begin) * line1.Substitute(segment.end);
  if (left > 0 || right > 0) {
    return std::nullopt;
  }
  return intersection.value();
}

template <typename NumberType>
template <bool IsClockwise, template <typename, typename> typename ContainerType, typename Alloc>
std::optional<Point<NumberType>> Segment<NumberType>::Intersection(const Polygon<
    NumberType,
    IsClockwise,
    ContainerType,
    Alloc>& polygon) const {
  throw std::invalid_argument("Is not implemented");
}

template <typename NumberType, bool IsClockwise, template <typename, typename> typename ContainerType, typename Alloc>
std::optional<Point<NumberType>> Polygon<NumberType,
                                         IsClockwise,
                                         ContainerType,
                                         Alloc>::Intersection(
    const Point<NumberType>& point) const {
  if (CheckPoint(point) == PointIntersection::Outside) {
    return std::nullopt;
  }
  return point;
}

template <typename NumberType, bool IsClockwise, template <typename, typename> typename ContainerType, typename Alloc>
std::optional<Point<NumberType>> Polygon<NumberType,
                                         IsClockwise,
                                         ContainerType,
                                         Alloc>::Intersection(
    const Line<NumberType>& line) const {
  throw std::invalid_argument("Is not implemented");
}

template <typename NumberType, bool IsClockwise, template <typename, typename> typename ContainerType, typename Alloc>
std::optional<Point<NumberType>> Polygon<NumberType,
                                         IsClockwise,
                                         ContainerType,
                                         Alloc>::Intersection(
    const Ray<NumberType>& ray) const {
  throw std::invalid_argument("Is not implemented");
}

template <typename NumberType, bool IsClockwise,
    template <typename, typename> typename ContainerType,
    typename Alloc>
template <bool is_const>
class Polygon<NumberType, IsClockwise, ContainerType, Alloc>::Iterator {
 public:
  using value_type =  // NOLINT
      std::conditional_t<is_const, const typename Container::value_type,
                         typename Container::value_type>;
  using pointer = value_type*;    // NOLINT
  using reference = value_type&;  // NOLINT
  using iterator_category =       // NOLINT
      typename Container::iterator::iterator_category;
  using difference_type =  // NOLINT
      typename Container::iterator::difference_type;

  friend Polygon;

  Iterator(container_iterator iterator)
      : iterator_(iterator) {}

  Iterator(const Iterator<is_const>& iterator)
      : iterator_(iterator.iterator_) {}

  Iterator(Iterator<is_const>&& iterator) noexcept
      : iterator_(std::move(iterator.iterator_)) {}

  Iterator& operator=(const Iterator<is_const>& iterator) {
    iterator_ = iterator.iterator_;
    return *this;
  }

  Iterator& operator=(Iterator<is_const>&& iterator) noexcept {
    if (&iterator != this) {
      iterator_ = std::move(iterator.iterator_);
    }
    return *this;
  }

  ~Iterator() = default;

  operator const_iterator() {
    return Polygon<NumberType,
                   IsClockwise,
                   ContainerType,
                   Alloc>::Iterator<true>(iterator_);
  };

  Iterator& operator++() {
    ++iterator_;
    return *this;
  }

  Iterator operator++(int)& {
    auto copy = *this;
    ++iterator_;
    return copy;
  }

  Iterator& operator--() {
    --iterator_;
    return *this;
  }

  Iterator operator--(int)& {
    auto copy = *this;
    --iterator_;
    return copy;
  }

  Iterator& operator+=(difference_type diff) {
    iterator_ += diff;
    return *this;
  }

  Iterator& operator-=(difference_type diff) {
    iterator_ -= diff;
    return *this;
  }

  Iterator operator+(difference_type diff) const {
    auto copy = *this;
    copy += diff;
    return copy;
  }

  Iterator operator-(difference_type diff) const {
    auto copy = *this;
    copy -= diff;
    return copy;
  }

  difference_type operator-(const Iterator<is_const>& other) const {
    return iterator_ - other.iterator_;
  }

  bool operator<(const Iterator<is_const>& other) const {
    return iterator_ < other.iterator_;
  }

  template <bool is_const_other>
  bool operator==(const Iterator<is_const_other>& other) const {
    return iterator_ == other.iterator_;
  }

  template <bool is_const_other>
  bool operator!=(const Iterator<is_const_other>& other) const {
    return !(*this == other);
  }

  bool operator<=(const Iterator<is_const>& other) {
    return (*this == other || *this < other);
  }

  bool operator>(const Iterator<is_const>& other) { return !(*this <= other); }

  bool operator>=(const Iterator<is_const>& other) { return !(*this < other); }

  pointer operator->() const { return &(*iterator_); }

  reference operator*() const { return *iterator_; }

 private:
  typename Container::iterator iterator_;
};

template <typename NumberType>
bool operator<(const Point<NumberType>& first, const Point<NumberType>& second) {
  if (first.x == second.x) {
    return first.y < second.y;
  }
  return first.x < second.x;
}


template <typename NumberType>
class ConvexHull {
 public:
  using PointType = Point<NumberType>;
  using VectorType = Vector<NumberType>;
  ConvexHull() = default;
  ConvexHull(const ConvexHull& convex_hull) = default;
  ConvexHull(ConvexHull&& convex_hull) noexcept = default;
  ConvexHull& operator=(const ConvexHull& convex_hull) = default;
  ConvexHull& operator=(ConvexHull&& convex_hull) noexcept = default;

  void Insert(const PointType& point) {
    if (size_ == 0) {
      upper_.insert(point);
      lower_.insert(point);
      ++size_;
      return;
    }
    if (size_ == 1) {
      upper_.insert(point);
      lower_.insert(point);
      ++size_;
      return;
    }
    if (IsInside(point)) {
      return;
    }
    ++size_;
    auto upper_begin = upper_.begin();
    auto upper_end = upper_.end();
    --upper_end;
    if (point.x < upper_begin->x) {
      ClearUpperRight(point, upper_.begin());
      ClearLowerRight(point, lower_.begin());
      InsertUp(point);
      InsertDown(point);
      CorrectSides();
      return;
    }
    if (point.x > upper_end->x) {
      ClearUpperLeft(point, --upper_.end());
      ClearLowerLeft(point, --lower_.end());
      InsertUp(point);
      InsertDown(point);
      CorrectSides();
      return;
    }
    auto upper_bound = upper_.upper_bound(point);
    if (upper_bound == upper_.begin() && point.y < upper_bound->y && size_ == 3) {
      upper_.insert(point);
      lower_.insert(point);
      return;
    }
    auto prev = std::next(upper_bound, -1);
    auto vertical = VectorType(0, -1);
    auto vector_from = VectorType(point, *prev);
    auto vector_to = VectorType(point, *upper_bound);
    if (VectorBetweenTwoRelative(vector_from, vector_to, vertical)) {
      CleanAndInsertUpper(point);
    } else {
      CleanAndInsertLower(point);
    }
    CorrectSides();
  }

  bool IsInside(const PointType& point) const {
    auto left_most = *upper_.begin();
    auto right_most = *--upper_.end();
    if (left_most.x > point.x || right_most.x < point.x) {
      return false;
    }
    auto location = PointGeolocation(point, Segment<NumberType>(left_most, right_most));
    if (location == On) {
      return true;
    }
    if (location == Up) {
      return FindUp(point);
    }
    if (location == Down) {
      return FindDown(point);
    }
    throw("Ops");
  }

 private:
  struct UpperComparator {
    bool operator()(const PointType& first, const PointType& second) const {
      if (first.x == second.x) {
        return first.y < second.y;
      }
      return first.x < second.x;
    }
  };

  struct LowerComparator {
    bool operator()(const PointType& first, const PointType& second) const {
      if (first.x == second.x) {
        return first.y > second.y;
      }
      return first.x < second.x;
    }
  };

  using UpperSetType = std::set<PointType>;
  using LowerSetType = std::set<PointType>;
  using UpperIterator = typename UpperSetType::iterator;
  using LowerIterator = typename LowerSetType::iterator;

  void CleanAndInsertUpper(const PointType& point) {
    auto end = *(--upper_.end());
    if (point.x == end.x) {
      auto prev = std::next(upper_.end(), -2);
      ClearUpperLeft(point, prev);
      upper_.erase(--upper_.end());
      InsertUp(point);
      InsertDown(point);
      return;
    }
    auto right = upper_.upper_bound(point);
    if (right != upper_.end()) {
      ClearUpperRight(point, right);
    }
    auto left = upper_.lower_bound(point);
    if (left != upper_.begin() && left != upper_.end()) {
      --left;
      ClearUpperLeft(point, left);
    }
    InsertUp(point);
  }

  void CleanAndInsertLower(const PointType& point) {
    auto begin = *(lower_.begin());
    if (point.x == begin.x) {
      auto next = std::next(lower_.begin(), 1);
      ClearLowerRight(point, next);
      lower_.erase(lower_.begin());
      InsertDown(point);
      InsertUp(point);
      return;
    }
    auto right = lower_.upper_bound(point);
    if (right != lower_.end()) {
//      if (right == lower_.begin()) {
//        ++right;
//      }
      ClearLowerRight(point, right);
    }
    auto left = lower_.lower_bound(point);
    if (left != lower_.begin() && left != lower_.end()) {
      --left;
      ClearLowerLeft(point, left);
    }
    InsertDown(point);
  }

  void CorrectSides() {
    return;
    CorrectLeft();
    CorrectRight();
  }

  void CorrectLeft() {
    auto left_most = lower_.begin();
    auto down = std::next(left_most, 1);
    auto up = std::next(upper_.begin(), 1);
    if (!Segment<NumberType>(*down, *up).Intersection(*left_most)) {
      return;
    }
    lower_.erase(lower_.begin());
    upper_.erase(upper_.begin());
    lower_.insert(lower_.begin(), *upper_.begin());
  }

  void CorrectRight() {
    auto right_most = *--lower_.end();
    auto down = *--(--lower_.end());
    auto up = *--(--upper_.end());
    if (!Segment<NumberType>(down, up).Intersection(right_most)) {
      return;
    }
    lower_.erase(--lower_.end());
    upper_.erase(--upper_.end());
    lower_.insert(lower_.end(), *--upper_.end());
  }


  void InsertUp(const PointType& point) {
    auto key_value = upper_.insert(point);
    if (!key_value.second) {
      throw(std::invalid_argument("point already exist"));
      return;
    }
    auto inserted_iterator = key_value.first;
    if (std::distance(inserted_iterator, upper_.end()) > 2) {
      auto next_iterator = std::next(inserted_iterator, 1);
      auto next_next_iterator = std::next(inserted_iterator, 2);
      if (Segment<NumberType>(*inserted_iterator, *next_next_iterator).Intersection(*next_iterator)) {
        upper_.erase(next_iterator);
      }
    }
    if (std::distance(upper_.begin(), inserted_iterator) > 1) {
      auto prev_iterator = std::next(inserted_iterator, -1);
      auto prev_prev_iterator = std::next(inserted_iterator, -2);
      if (Segment<NumberType>(*inserted_iterator, *prev_prev_iterator).Intersection(*prev_iterator)) {
        upper_.erase(prev_iterator);
      }
    }
    if (std::distance(upper_.begin(), inserted_iterator) > 0 &&
        std::distance(inserted_iterator, upper_.end()) > 1) {
      auto prev_iterator = std::next(inserted_iterator, -1);
      auto next_iterator = std::next(inserted_iterator, 1);
      if (Segment<NumberType>(*prev_iterator, *next_iterator).Intersection(*inserted_iterator)) {
        upper_.erase(inserted_iterator);
      }
    }
  }

  void InsertDown(const PointType& point) {
    auto key_value = lower_.insert(point);
    if (!key_value.second) {
      throw(std::invalid_argument("point already exist"));
      return;
    }
    auto inserted_iterator = key_value.first;
    if (std::distance(inserted_iterator, lower_.end()) > 2) {
      auto next_iterator = std::next(inserted_iterator, 1);
      auto next_next_iterator = std::next(inserted_iterator, 2);
      if (Segment<NumberType>(*inserted_iterator, *next_next_iterator).Intersection(*next_iterator)) {
        lower_.erase(next_iterator);
      }
    }
    if (std::distance(lower_.begin(), inserted_iterator) > 1) {
      auto prev_iterator = std::next(inserted_iterator, -1);
      auto prev_prev_iterator = std::next(inserted_iterator, -2);
      if (Segment<NumberType>(*inserted_iterator, *prev_prev_iterator).Intersection(*prev_iterator)) {
        lower_.erase(prev_iterator);
      }
    }
    if (std::distance(lower_.begin(), inserted_iterator) > 0 &&
        std::distance(inserted_iterator, lower_.end()) > 1) {
      auto prev_iterator = std::next(inserted_iterator, -1);
      auto next_iterator = std::next(inserted_iterator, 1);
      if (Segment<NumberType>(*prev_iterator, *next_iterator).Intersection(*inserted_iterator)) {
        lower_.erase(inserted_iterator);
      }
    }
  }

  void ClearUpperRight(const PointType& point, UpperIterator iterator) {
    while (iterator != --upper_.end() && iterator != upper_.end() && upper_.size() > 1) {
      auto next_iterator = iterator;
      ++next_iterator;
      auto cur = *iterator;
      auto next = *next_iterator;
      auto vector_to_next = Vector<NumberType>(cur, next);
      auto vector_to_point = Vector<NumberType>(cur, point);
      auto mul = VectorMul(vector_to_next, vector_to_point);
      if (Round(mul) >= 0) {
        upper_.erase(iterator);
        iterator = next_iterator;
        continue;
      }
      break;
    }
  }

  void ClearUpperLeft(const PointType& point, UpperIterator iterator) {
    while (iterator != upper_.begin() && upper_.size() > 1) {
      auto prev_iterator = iterator;
      --prev_iterator;
      auto cur = *iterator;
      auto prev = *(prev_iterator);
      auto vector_to_prev = Vector<NumberType>(cur, prev);
      auto vector_to_point = Vector<NumberType>(cur, point);
      auto mul = VectorMul(vector_to_prev, vector_to_point);
      if (Round(mul) <= 0) {
        upper_.erase(iterator);
        iterator = prev_iterator;
        continue;
      }
      break;
    }
  }

  void ClearLowerLeft(const PointType& point, LowerIterator iterator) {
    if (iterator == lower_.end()) {
      return;
    }
    while (iterator != lower_.begin() && lower_.size() > 1) {
      auto prev_iterator = iterator;
      --prev_iterator;
      auto cur = *iterator;
      auto prev = *(prev_iterator);
      auto vector_to_prev = Vector<NumberType>(cur, prev);
      auto vector_to_point = Vector<NumberType>(cur, point);
      if (Round(VectorMul(vector_to_prev, vector_to_point)) >= 0) {
        lower_.erase(iterator);
        iterator = prev_iterator;
        continue;
      }
      break;
    }
  }

  void ClearLowerRight(const PointType& point, LowerIterator iterator) {
    while (iterator != --lower_.end() && lower_.size() > 1) {
      auto next_iterator = iterator;
      ++next_iterator;
      auto cur = *iterator;
      auto next = *next_iterator;
      auto vector_to_next = VectorType(cur, next);
      auto vector_to_point = VectorType(cur, point);
      auto mul = VectorMul(vector_to_next, vector_to_point);
      if (Round(mul) <= 0) {
        lower_.erase(iterator);
        iterator = next_iterator;
        continue;
      }
      break;
    }

  }

  bool FindUp(const PointType& point) const {
    auto right = upper_.upper_bound(point);
    if (right == upper_.end()) {
      return false;
    }
    if (right == upper_.begin()) {
      if (point == *right) {
        return true;
      }
      return false;
    }
    auto left = std::next(right, -1);
    auto tmp = PointGeolocation(point, Segment<NumberType>(*left, *right));
    if (tmp == Up || tmp == Vertical) {
      return false;
    }
    return true;
  }

  bool FindDown(const PointType& point) const {
    auto right = lower_.upper_bound(point);
    if (right == lower_.end()) {
      return false;
    }
    if (right == lower_.begin()) {
      if (point == *right) {
        return true;
      }
      return false;
    }
    auto left = std::next(right, -1);
    auto tmp = PointGeolocation(point, Segment<NumberType>(*left, *right));
    if (tmp == Down || tmp == Vertical) {
      return false;
    }
    return true;
  }

  enum PointGeolocationType {
    Up, On, Down, Vertical
  };

  PointGeolocationType PointGeolocation(const PointType& point, const Segment<NumberType>& segment) const {
    if (segment.Intersection(point)) {
      return On;
    }
    auto vertical = VectorType(0, -1);
    auto mul = VectorMul(segment.GetGuiding(), vertical);
    if (Round(mul) == 0) {
      return Vertical;
    }
    auto vector_to = VectorType(point, segment.begin);
    auto vector_from = VectorType(point, segment.end);
    if (VectorBetweenTwoRelative(vector_from, vector_to, vertical)) {
      return Up;
    }
    return Down;
  }

  bool VectorBetweenTwoRelative(const VectorType& from,
                                const VectorType& to,
                                const VectorType& between) const {
    bool left = sgn(VectorMul(from, between))
        * sgn(VectorMul(from, to)) >= 0;
    bool right =
        sgn(VectorMul(to, between)) * sgn(VectorMul(to, from))
            >= 0;
    return (left && right);
  }

  UpperSetType upper_;
  LowerSetType lower_;
  size_t size_ = 0;
};

template <typename Geometry1, typename Geometry2>
bool Intersection(const Geometry1& geometry_1, const Geometry2& geometry_2) {
  if (geometry_1.Intersection(geometry_2)) {
    return true;
  }
  return false;
}

template <typename NumberType>
class Solver {
 public:
  void Read() {
    std::cin >> q;
  }

  void SolveAndPrint() {
    ConvexHull<NumberType> convex_hull;
    size_t command = 2;
    Point<NumberType> point;
    for (size_t i = 0; i < q; ++i) {
      std::cin >> command >> point;
      if (command == 1) {
        convex_hull.Insert(point);
        continue;
      }
      if (convex_hull.IsInside(point)) {
        std::cout << "YES\n";
        continue;
      }
      std::cout << "NO\n";
    }
    return;
  }

 private:
  int64_t q = 0;
};

int main() {
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(7);
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.tie(nullptr);
  Solver<int64_t> solver;
  solver.Read();
  solver.SolveAndPrint();
}
