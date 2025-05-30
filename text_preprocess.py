import re


def convert_unicode(text):
    char1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    char1252 = char1252.split("|")
    charutf8 = charutf8.split("|")

    dic = {}
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return re.sub(
        r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
        lambda x: dic[x.group()],
        text,
    )


class TextNormalize:
    def __init__(self):
        self.vowels_to_ids = {}
        self.vowels_table = [
            ["a", "à", "á", "ả", "ã", "ạ", "a"],
            ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
            ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
            ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
            ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
            ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
            ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
            ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "o"],
            ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
            ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
            ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
            ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
        ]
        pass

    def createVowelsTable(self):
        """Create Vowels Table"""
        for i in range(len(self.vowels_table)):
            for j in range(len(self.vowels_table[i]) - 1):
                self.vowels_to_ids[self.vowels_table[i][j]] = (i, j)

    def IsValidVietnameseWord(self, word):
        """Nguyên âm chỉ có thể đứng chung với nguyên âm. Một từ không thể có 2 nguyên âm cách nhau bởi 1 phụ âm"""
        chars = list(word)
        # nguyen am
        vowel_index = -1
        for i in range(len(chars)):
            idx_vowel_table = self.vowels_to_ids.get(chars[i], (-1, -1))[0]
            if idx_vowel_table != -1:
                if vowel_index == -1:
                    vowel_index = i
                else:
                    if i - vowel_index != 1:
                        return False
                    vowel_index = i
        return True

    def WordStandardized(self, word):
        """Standardize Word"""
        if not self.IsValidVietnameseWord(word):
            return word

        chars = list(word)
        vowel_indexes = []

        # tìm vị trí nguyên âm
        qu_or_gi = False
        thanh_dieu = 0
        for i in range(len(chars)):
            vowel_table_row, vowel_table_col = self.vowels_to_ids.get(
                chars[i], (-1, -1)
            )
            if vowel_table_row == -1:
                continue
            # qu
            if vowel_table_row == 9:
                if i != 0 and chars[i - 1] == "q":
                    chars[i] = "u"
                    qu_or_gi = True
            # gi
            elif vowel_table_row == 5:
                if i != 0 and chars[i - 1] == "g":
                    chars[i] = "i"
                    qu_or_gi = True

            # có chứa thanh điệu
            if vowel_table_col != 0:
                thanh_dieu = vowel_table_col
                chars[i] = self.vowels_table[vowel_table_row][0]

            vowel_indexes.append(i)
        # 1 nguyên âm
        if len(vowel_indexes) == 1:
            c = chars[vowel_indexes[0]]
            chars[vowel_indexes[0]] = self.vowels_table[self.vowels_to_ids[c][0]][
                thanh_dieu
            ]
            return "".join(chars)

        for idx_vowel in vowel_indexes:
            vowel_table_row, vowel_table_col = self.vowels_to_ids.get(
                chars[idx_vowel], (-1, -1)
            )
            # ê, ơ, ô
            if vowel_table_row == 4 or vowel_table_row == 7 or vowel_table_row == 8:
                c = chars[idx_vowel]
                chars[idx_vowel] = self.vowels_table[self.vowels_to_ids[c][0]][
                    thanh_dieu
                ]
                return "".join(chars)

            # kiểm tra qu và gi, 2-3 nguyên âm thì nguyên âm thứ 2 chứa dấu
            if qu_or_gi:
                if len(vowel_indexes) == 2 or len(vowel_indexes) == 3:
                    c = chars[vowel_indexes[1]]
                    chars[vowel_indexes[1]] = self.vowels_table[
                        self.vowels_to_ids[c][0]
                    ][thanh_dieu]
                return "".join(chars)

            # 2 nguyên âm
            if len(vowel_indexes) == 2:
                # âm cuối là nguyên âm
                if vowel_indexes[-1] == len(chars) - 1:
                    c = chars[vowel_indexes[0]]
                    chars[vowel_indexes[0]] = self.vowels_table[
                        self.vowels_to_ids[c][0]
                    ][thanh_dieu]
                else:
                    c = chars[vowel_indexes[-1]]
                    chars[vowel_indexes[-1]] = self.vowels_table[
                        self.vowels_to_ids[c][0]
                    ][thanh_dieu]
                return "".join(chars)

            elif len(vowel_indexes) == 3:
                # âm cuối là nguyên âm
                if vowel_indexes[-1] == len(chars) - 1:
                    c = chars[vowel_indexes[1]]
                    chars[vowel_indexes[1]] = self.vowels_table[
                        self.vowels_to_ids[c][0]
                    ][thanh_dieu]
                else:
                    c = chars[vowel_indexes[-1]]
                    chars[vowel_indexes[-1]] = self.vowels_table[
                        self.vowels_to_ids[c][0]
                    ][thanh_dieu]
                return "".join(chars)

        return "".join(chars)

    def normalize(self, text):

        # Chuyen sang viet thuong
        text = text.lower()
        text = convert_unicode(text)

        text = [self.WordStandardized(word) for word in text.split()]
        text = " ".join(text)

        # Rút gọn từ kéo dài
        text = re.sub(r"(\w)\1+", r"\1", text)

        # remove hastag
        text = re.sub("(@[A-Za-z0-9]+)|(#[0-9A-Za-z]+)", "", text)

        # xóa space d
        text = re.sub(r"( )\1+", r"\1", text)
        # Add space around special characters
        text = re.sub(r"([.@/,;:!?()\[\]{}<>|\\+=\-_*&^%$#\"\'`~])", r" \1 ", text)
        # Clean up multiple spaces
        text = re.sub(r"\s+", " ", text.strip())
        text = text.replace("“", "")

        return text


if __name__ == "__main__":
    text_normalizer = TextNormalize()
    text_normalizer.createVowelsTable()
    sample_text = "Hà Nội là thủ đô của Việt Nam hả? @nguyenvana #vietnamexample"
    normalized_text = text_normalizer.normalize(sample_text)
    print(normalized_text)  # Output: "ha noi la thu do cua viet nam"
