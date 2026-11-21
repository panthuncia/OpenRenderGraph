// Recording tags keyed per extension (RecordingTags, RecordingContext::Tag).
//
// Several extensions tag one graph, each under its own key. A set never changes once shared: setting a tag makes a new set,
// so a recording keeps the values it was recorded with.

#include "Render/PreparedPass.h"
#include "Render/RecordingTags.h"
#include "TestSupport.h"

#include <cstdio>
#include <memory>

namespace {
const org::RecordingTagKey kFirst{"first"};
const org::RecordingTagKey kSecond{"second"};

int TestKeys() {
    const auto one = std::make_shared<const int>(1);
    const auto two = std::make_shared<const int>(2);
    const org::RecordingTags empty;
    CHECK(empty.Empty());
    CHECK(empty.Find(kFirst) == nullptr);
    const auto first = empty.With(kFirst, one);
    const auto both = first.With(kSecond, two);
    CHECK(both.Size() == 2);
    CHECK(both.Find(kFirst) == one.get());
    CHECK(both.Find(kSecond) == two.get());
    // The set it was made from is unchanged.
    CHECK(first.Size() == 1 && first.Find(kSecond) == nullptr);
    // Two keys with the same name are still two keys.
    const org::RecordingTagKey sameName{"first"};
    CHECK(both.Find(sameName) == nullptr);
    return 0;
}

int TestReplaceAndRemove() {
    const auto one = std::make_shared<const int>(1);
    const auto three = std::make_shared<const int>(3);
    const auto tags = org::RecordingTags{}.With(kFirst, one).With(kSecond, one);
    const auto replaced = tags.With(kFirst, three);
    CHECK(replaced.Size() == 2);
    CHECK(replaced.Find(kFirst) == three.get());
    CHECK(replaced.Find(kSecond) == one.get());
    const auto removed = replaced.With(kFirst, nullptr);
    CHECK(removed.Size() == 1 && removed.Find(kFirst) == nullptr && removed.Find(kSecond) == one.get());
    return 0;
}

int TestRecordingContext() {
    const auto one = std::make_shared<const int>(1);
    auto context = org::RecordingContext::ForSkipQuestions(org::RunRef{});
    CHECK(context.Tag(kFirst) == nullptr);
    auto tags = std::make_shared<const org::RecordingTags>(org::RecordingTags{}.With(kFirst, one));
    context.SetTags(tags);
    tags.reset();
    // The context holds the set and its values.
    CHECK(context.Tag(kFirst) == one.get());
    CHECK(context.Tag(kSecond) == nullptr);
    return 0;
}
}

int main() {
    if (const int line = TestKeys()) return line;
    if (const int line = TestReplaceAndRemove()) return line;
    if (const int line = TestRecordingContext()) return line;
    std::puts("OpenRenderGraphRecordingTagTests passed");
    return 0;
}
