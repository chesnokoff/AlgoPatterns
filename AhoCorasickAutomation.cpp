#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

template <typename Container>
class AlphaBet {
 public:
  template <bool is_const>
  class Iterator;
  using iterator = Iterator<false>;                                // NOLINT
  using const_iterator = Iterator<true>;                           // NOLINT
  using reverse_iterator = std::reverse_iterator<iterator>;        // NOLINT
  using container_allocator = typename Container::allocator_type;  // NOLINT
  using value_type = typename Container::value_type;               // NOLINT
  using container = Container;                                     // NOLINT

  AlphaBet() = default;

  explicit AlphaBet(const Container& container)
      : container_(container), alphabet_power_(container.size()) {}

  AlphaBet(const AlphaBet<Container>& alpha_bet)
      : container_(alpha_bet.container_),
        alphabet_power_(alpha_bet.alphabet_power_) {}

  AlphaBet(AlphaBet<Container>&& alpha_bet) noexcept
      : container_(std::move(alpha_bet.container_)),
        alphabet_power_(alpha_bet.alphabet_power_) {}

  AlphaBet& operator=(const AlphaBet<Container>& alpha_bet) {
    container_ = alpha_bet.container_;
    alphabet_power_ = alpha_bet.alphabet_power_;
    return *this;
  }

  AlphaBet& operator=(AlphaBet<Container>&& alpha_bet) noexcept {
    if (&alpha_bet != this) {
      container_ = std::move(alpha_bet.container_);
      alphabet_power_ = alpha_bet.alphabet_power_;
      alpha_bet.alphabet_power_ = alpha_bet.container_.size();
    }
    return *this;
  }

  value_type GetSymbol(size_t pos) { return container_[pos]; }

  size_t GetPosOfSymbol(const value_type& value) {
    for (size_t pos = 0; pos < container_.size(); ++pos) {
      if (container_[pos] == value) {
        return pos;
      }
    }
    return container_.size();
  }

  size_t Power() const { return alphabet_power_; }

  iterator begin() { return iterator(container_.begin()); }  // NOLINT

  const_iterator begin() const {  // NOLINT
    return const_iterator(container_.begin());
  }

  reverse_iterator rbegin() { return reverse_iterator(begin()); }  // NOLINT

  iterator end() { return iterator(container_.end()); }  // NOLINT

  const_iterator end() const {  // NOLINT
    return const_iterator(container_.end());
  }

  reverse_iterator rend() { return reverse_iterator(end()); }  // NOLINT

 private:
  Container container_;
  size_t alphabet_power_;
};

template <typename Container>
template <bool is_const>
class AlphaBet<Container>::Iterator {
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

  Iterator(const typename Container::iterator& iterator)
      : iterator_(iterator) {}

  Iterator(const Iterator<is_const>& iterator)
      : iterator_(iterator.iterator_) {}

  Iterator(Iterator<is_const>&& iterator)
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
    return AlphaBet<Container>::Iterator<true>(iterator_);
  };

  Iterator& operator++() {
    ++iterator_;
    return *this;
  }

  Iterator operator++(int) {
    auto copy = *this;
    ++iterator_;
    return copy;
  }

  Iterator& operator--() {
    --iterator_;
    return *this;
  }

  Iterator operator--(int) {
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

template <typename AlphaBet>
class AhoCorasickAutomation {
 public:
  using value_type = typename AlphaBet::value_type;  // NOLINT

  explicit AhoCorasickAutomation(AlphaBet alpha_bet)
      : alpha_bet_(std::move(alpha_bet)) {}

  void Add(const std::vector<value_type>& word) {
    if (trie_.empty()) {
      trie_.emplace_back(Node{alpha_bet_.Power()});
    }
    size_t node = 0;
    for (const auto& symbol : word) {
      if (trie_[node].to[alpha_bet_.GetPosOfSymbol(symbol)] == -1) {
        trie_[node].to[alpha_bet_.GetPosOfSymbol(symbol)] =
            static_cast<ssize_t>(trie_.size());
        trie_.emplace_back(Node{alpha_bet_.Power()});
        ++(trie_[trie_.size() - 1].len = trie_[node].len);
      }
      node = trie_[node].to[alpha_bet_.GetPosOfSymbol(symbol)];
    }
    trie_[node].terminal = true;
    trie_[node].dict_pos.push_back(dict_size_);
    ++dict_size_;
  }

  void Init() {
    Resize();
    std::queue<size_t> queue;
    queue.push(0);
    PreCalculateGoForRoot();
    while (!queue.empty()) {
      size_t node = queue.front();
      queue.pop();
      CalculateCompressedLink(node);
      for (size_t edge = 0; edge < alpha_bet_.Power(); ++edge) {
        ssize_t next = trie_[node].to[edge];
        if (next == -1) {
          continue;
        }
        link_[next] = node > 0 ? go_[link_[node]][edge] : 0;
        for (size_t next_symbol = 0; next_symbol < alpha_bet_.Power();
             ++next_symbol) {
          if (trie_[next].to[next_symbol] != -1) {
            go_[next][next_symbol] = trie_[next].to[next_symbol];
          } else {
            go_[next][next_symbol] = go_[link_[next]][next_symbol];
          }
        }
        queue.push(next);
      }
    }
  }

  auto Calculate(const std::vector<value_type>& text) {
    size_t node = 0;
    for (size_t pos = 0; pos < text.size(); ++pos) {
      node = go_[node][alpha_bet_.GetPosOfSymbol(text[pos])];
      CalculateAnswerForTerminals(node, pos);
    }
    return dict_pos_;
  }

  void Reset() { dict_pos_.resize(dict_size_, std::vector<size_t>(1)); }

 private:
  struct Node {
    size_t alphabet_power;
    std::vector<ssize_t> to = std::vector<ssize_t>(alphabet_power, -1);
    bool terminal = false;
    size_t len = 0;
    std::vector<size_t> dict_pos = std::vector<size_t>();
  };

  void Resize() {
    dict_pos_.resize(dict_size_, std::vector<size_t>(1));
    link_.resize(trie_.size());
    compressed_link_.resize(trie_.size());
    go_.resize(trie_.size(), std::vector<ssize_t>(alpha_bet_.Power()));
  }

  void PreCalculateGoForRoot() {
    const auto& root = trie_[0];
    for (size_t symbol = 0; symbol < alpha_bet_.Power(); ++symbol) {
      if (root.to[symbol] != -1) {
        go_[0][symbol] = root.to[symbol];
      }
    }
  }

  void CalculateCompressedLink(size_t node) {
    auto link_node = link_[node];
    if (trie_[link_node].terminal) {
      compressed_link_[node] = link_node;
      return;
    }
    compressed_link_[node] = compressed_link_[link_node];
  }

  void CalculateAnswerForTerminals(size_t node, size_t pos) {
    while (node != 0) {
      if (trie_[node].terminal) {
        for (const auto& dict_pos_node : trie_[node].dict_pos) {
          ++dict_pos_[dict_pos_node][0];
          dict_pos_[dict_pos_node].push_back(pos + 1 - trie_[node].len);
        }
      }
      node = compressed_link_[node];
    }
  }

  AlphaBet alpha_bet_;
  std::vector<Node> trie_;
  std::vector<ssize_t> link_;
  std::vector<ssize_t> compressed_link_;
  std::vector<std::vector<ssize_t>> go_;
  std::vector<std::vector<size_t>> dict_pos_;
  size_t dict_size_ = 0;
};

class Solver {
 public:
  void Read() {
    std::string text;
    std::cin >> text;
    text_ = std::vector<char>(text.begin(), text.end());
    size_t dict_size;
    std::cin >> dict_size;
    std::string pattern;
    for (size_t i = 0; i < dict_size; ++i) {
      std::cin >> pattern;
      std::vector<char> word(pattern.begin(), pattern.end());
      automation_.Add(word);
    }
  }

  void Solve() {
    automation_.Init();
    answer_ = automation_.Calculate(text_);
  }

  void Print() {
    for (auto& row : answer_) {
      std::cout << row[0] << ' ';
      for (size_t pos = 1; pos < row.size(); ++pos) {
        std::cout << row[pos] + 1 << ' ';
      }
      std::cout << std::endl;
    }
  }

 private:
  static AlphaBet<std::deque<char>> GenerateAlphabet() {
    std::deque<char> container('z' - 'a' + 1);
    for (size_t pos = 0; pos < container.size(); ++pos) {
      container[pos] = 'a' + pos;
    }
    return AlphaBet<std::deque<char>>(container);
  }

  AlphaBet<std::deque<char>> alpha_bet_ = GenerateAlphabet();
  AhoCorasickAutomation<AlphaBet<std::deque<char>>> automation_ =
      AhoCorasickAutomation<AlphaBet<std::deque<char>>>(alpha_bet_);
  std::vector<char> text_;
  std::vector<std::vector<size_t>> answer_;
};

int main() {
  Solver solver;
  solver.Read();
  solver.Solve();
  solver.Print();
}
