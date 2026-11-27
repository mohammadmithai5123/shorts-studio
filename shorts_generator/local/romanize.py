"""Hindi (Devanagari) and Urdu script written in Roman letters: Hinglish / Roman Urdu, the way people type it.

This changes the letters only, never the language: "मुझे नहीं पता" -> "mujhe nahi pata". English words that the
captions spelled in Hindi letters get their English spelling back ("वीडियो" -> "video").

A word is looked up in this order:
  1. the built-in list of common words (consistent spellings, no network);
  2. answers the AI gave before (``<output>/cache/roman_words.json``); :func:`prepare` asks it for the unknown ones
     in one time-limited batch;
  3. spelling rules (always works; weakest for rare words and for Urdu, which leaves short vowels out).
Timings are untouched: one caption word in, one caption word out.
"""
import json
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config

LLM_SECONDS = 60          # a busy AI service must not hold up the render: the list and the rules cover for it
CHUNK_WORDS = 250

_DEVA = r"ऀ-ॣ०-ॿ‌‍"
_ARAB = r"ء-ٟٮ-ۓە-ۭۺ-ۿݐ-ݿﭐ-﷿ﹰ-ﻼ"
_RUN = re.compile(f"[{_DEVA}]+|[{_ARAB}]+")
_ANY = re.compile(r"[ऀ-ॿ؀-ۿݐ-ݿﭐ-﷿ﹰ-ﻼ]")
_PUNCT = {"।": "", "॥": "", "۔": "", "،": ",", "؟": "?", "؛": ";"}
_DIGITS = str.maketrans("०१२३४५६७८९٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "012345678901234567890123456789")

# --------------------------------------------------------------------------------------------- #
# Common words ("word:roman", separated by spaces)
# --------------------------------------------------------------------------------------------- #
_HINDI = """
है:hai हैं:hain हूं:hoon हो:ho था:tha थी:thi थे:the नहीं:nahi नही:nahi ना:na न:na मत:mat हां:haan क्या:kya क्यों:kyun क्यूं:kyun
ये:ye यह:yeh वो:wo वह:woh तो:toh की:ki कि:ki के:ke का:ka को:ko से:se में:mein पे:pe पर:par ने:ne भी:bhi ही:hi और:aur या:ya
लेकिन:lekin मगर:magar अगर:agar क्योंकि:kyunki इसलिए:isliye वरना:warna बल्कि:balki मैं:main हम:hum तू:tu तुम:tum आप:aap मुझे:mujhe मुझको:mujhko
तुझे:tujhe तुम्हें:tumhein हमें:humein आपको:aapko इसको:isko उसको:usko इनको:inko उनको:unko इन्हें:inhein उन्हें:unhein उन्होंने:unhone
मेरा:mera मेरी:meri मेरे:mere तेरा:tera तेरी:teri तेरे:tere हमारा:hamara हमारी:hamari हमारे:hamare आपका:aapka आपकी:aapki आपके:aapke
इसका:iska इसकी:iski इसके:iske उसका:uska उसकी:uski उसके:uske अपना:apna अपनी:apni अपने:apne मैंने:maine आपने:aapne उसने:usne हमने:humne
जिसने:jisne किसने:kisne किसी:kisi किसको:kisko कोई:koi कुछ:kuch कौन:kaun सब:sab सबको:sabko सबसे:sabse सारा:saara सारी:saari सारे:saare
इस:is उस:us इन:in एक:ek दो:do कम:kam ज्यादा:zyada ज़्यादा:zyada बहुत:bahut थोड़ा:thoda थोड़ी:thodi बड़ा:bada बड़ी:badi बड़े:bade
अच्छा:accha अच्छी:acchi अच्छे:acche सही:sahi गलत:galat ठीक:theek बंद:band खत्म:khatam शुरू:shuru पूरा:poora पूरी:poori
अब:ab अभी:abhi कभी:kabhi जब:jab तब:tab फिर:phir फ़िर:phir बाद:baad पहले:pehle पहला:pehla पहली:pehli आज:aaj कल:kal रोज:roz रात:raat दिन:din
यहां:yahan वहां:wahan कहां:kahan जहां:jahan यही:yahi वही:wahi कहीं:kahin इधर:idhar उधर:udhar किधर:kidhar अंदर:andar बाहर:bahar
ऊपर:upar नीचे:neeche आगे:aage पीछे:peeche सामने:saamne साथ:saath पास:paas दूर:door तक:tak तरफ:taraf बीच:beech लिए:liye लिये:liye
वाला:wala वाली:wali वाले:wale जो:jo जैसे:jaise वैसे:waise ऐसे:aise ऐसा:aisa ऐसी:aisi कैसे:kaise इतना:itna इतनी:itni कितना:kitna कितने:kitne
कितनी:kitni हर:har सिर्फ:sirf शायद:shayad बिल्कुल:bilkul बिलकुल:bilkul जल्दी:jaldi दोबारा:dobara वापस:wapas बार:baar मतलब:matlab
रहा:raha रही:rahi रहे:rahe रह:reh रहना:rehna रहता:rehta रहती:rehti रहते:rehte कर:kar करो:karo करें:karein करना:karna करने:karne करनी:karni
करता:karta करते:karte करके:karke करेंगे:karenge करूंगा:karunga करवा:karwa किया:kiya
गया:gaya गई:gayi गए:gaye जा:ja जाओ:jao जाना:jaana जाने:jaane जाता:jaata जाते:jaate जाए:jaye जाएगा:jayega जाएगी:jayegi जाएंगे:jayenge जाके:jaake
हुआ:hua हुई:hui हुए:hue होए:hoye होता:hota होती:hoti होना:hona होने:hone होगा:hoga होगी:hogi
आ:aa आया:aaya आए:aaye आओ:aao आता:aata आती:aati आएगा:aayega आएगी:aayegi आय:aay
दे:de दो:do दी:di दिया:diya देता:deta देते:dete दूं:doon दूंगा:dunga ले:le लो:lo ली:li लिया:liya लें:lein लेना:lena लेने:lene लेनी:leni लेते:lete लेके:leke
देख:dekh देखो:dekho देखा:dekha देखना:dekhna देखते:dekhte दिखा:dikha ढूंढ:dhoondh
बोल:bol बोला:bola बोलो:bolo बता:bata बताओ:batao बात:baat कह:keh कहा:kaha कहना:kehna कहते:kehte कहता:kehta कहती:kehti
चल:chal चलो:chalo चला:chala चली:chali चले:chale चलते:chalte निकल:nikal निकलो:niklo रुक:ruk रुको:ruko भागो:bhaago पहुंच:pahunch
सकता:sakta सकती:sakti सकते:sakte चाहिए:chahiye चाह:chah पड़ेगा:padega पड़ना:padna पकड़:pakad फाड़:phaad रख:rakh मार:maar मारो:maaro मर:mar
लग:lag लगा:laga लगी:lagi लगे:lage लगता:lagta लगाओ:lagao बन:ban बना:bana बढ़ाने:badhane मिल:mil समझ:samajh पता:pata लिखा:likha उठा:utha छुपो:chhupo खेल:khel
भाई:bhai यार:yaar जी:ji अरे:arre अबे:abe ओए:oye ओ:o ओह:oh अ:a हे:hey साहब:sahab दोस्त:dost दोस्तों:doston लोग:log लोगों:logon
बंदा:banda बंदे:bande लड़का:ladka लड़की:ladki लड़के:ladke लड़कियां:ladkiyan बहन:behen मम्मी:mummy पापा:papa मां:maa बेटा:beta बेटी:beti
घर:ghar काम:kaam नाम:naam जगह:jagah चीज:cheez चीज़:cheez चीजें:cheezein चीज़ें:cheezein तरह:tarah लात:laat हाथ:haath नाक:naak दिल:dil जान:jaan
पैसा:paisa पैसे:paise पैसों:paison रुपए:rupaye रुपये:rupaye हजार:hazaar हज़ार:hazaar लाख:lakh करोड़:crore साल:saal महीना:mahina
प्यार:pyaar मोहब्बत:mohabbat इश्क:ishq शादी:shaadi ज़िंदगी:zindagi जिंदगी:zindagi दुनिया:duniya वक्त:waqt सच:sach झूठ:jhooth दर्द:dard खुश:khush
खुद:khud याद:yaad ख्याल:khayal मजा:maza मज़ा:maza पसंद:pasand वगैरह:wagairah जरा:zara ज़रा:zara जरूर:zaroor ज़रूर:zaroor मुश्किल:mushkil
आवाज:awaaz आवाज़:awaaz हंसी:hansi संगीत:sangeet प्रशंसा:prashansa हौसला:hausla चीखने:cheekhne गाना:gaana गाने:gaane गाड़ी:gaadi
पुलिस:police जेल:jail पानी:paani चाय:chai गोली:goli लूट:loot कपड़े:kapde जहाज:jahaaz जहाज़:jahaaz नजर:nazar नज़र:nazar फट:phat गांड:gaand
फूल:phool फल:phal फेंक:phenk फटाफट:phataphat बढ़िया:badhiya दीवार:deewar बस:bas सा:sa इसमें:ismein उसमें:usmein इनमें:inmein उनमें:unmein किसमें:kismein जिसमें:jismein इसी:isi देना:dena
कहानी:kahani शहर:shehar वाह:waah पक्का:pakka बच्चा:baccha बच्चे:bacche पागल:pagal कमाल:kamaal मज़ाक:mazaak मजाक:mazaak
"""

# English words the captions spell in Hindi letters
_ENGLISH = """
द:the ए:ae इज:is इज़:is आर:are वी:we एम:am वाज:was बी:be डू:do डन:done कैन:can विल:will हैव:have गेट:get गो:go लेट्स:let's यू:you योर:your
मी:me माय:my इट:it इट्स:it's दिस:this दैट:that व्हाट:what व्हाई:why वाई:why वेयर:where हाउ:how हियर:here नाउ:now जस्ट:just नो:no यस:yes
नॉट:not डोंट:don't एंड:and ऑन:on ऑफ:of ऑफ़:of एट:at इन:in फॉर:for फ्रॉम:from अप:up आउट:out बट:but सो:so ऑल:all मोर:more सम:some वन:one टू:to
बाय:bye अबाउट:about बैक:back अगेन:again ओवर:over राइट:right लेफ्ट:left साइड:side टॉप:top लास्ट:last फर्स्ट:first बेस्ट:best गुड:good
नाइस:nice रियल:real फुल:full वेरी:very मच:much ओके:okay हाय:hi हेलो:hello प्लीज:please प्लीज़:please थैंक:thank थैंक्स:thanks सॉरी:sorry
गाइस:guys गाइज:guys गाइज़:guys ब्रो:bro बेबी:baby मैन:man मिस्टर:mister सर:sir मैम:ma'am गॉड:god शिट:shit
थिंक:think वांट:want नीड:need लव:love लाइक:like लुक:look गिव:give टेक:take कीप:keep लीव:leave हेल्प:help वेट:wait चेक:check मूव:move
स्टार्ट:start स्टॉप:stop ओपन:open प्ले:play सेव:save कॉल:call चेंज:change पास:paas रेडी:ready फिक्स:fix अनलॉक:unlock
वीडियो:video चैट:chat लाइव:live स्ट्रीम:stream गेम:game लेवल:level मिशन:mission मैप:map क्लब:club ग्राफिक्स:graphics शो:show सीन:scene
इंट्रो:intro स्टोरी:story म्यूजिक:music डांस:dance स्टेज:stage जज:judge कॉमेडी:comedy फनी:funny क्रेजी:crazy क्रेज़ी:crazy
चैनल:channel सब्सक्राइब:subscribe कमेंट:comment शेयर:share यूट्यूब:YouTube गूगल:Google इंडिया:India सुपर:super
टाइम:time मिनट:minute मिनट्स:minutes सेकंड:second लाइफ:life स्टार:star लैंड:land मनी:money कैश:cash बैंक:bank बिलियन:billion अकाउंट:account
कार:car बाइक:bike ट्रक:truck टैक्सी:taxi प्लेन:plane जेट:jet एयरपोर्ट:airport बोट:boat रोड:road सिटी:city स्पीड:speed गैस:gas
गन:gun मिलिट्री:military हाउस:house स्टोर:store शॉपिंग:shopping पैकेज:package कार्ड:card बैग:bag कैप:cap कैप्स:caps पोल:pole बेल:bell
फेस:face रेड:red ग्रीन:green बियर:beer कॉफी:coffee हेल्थ:health स्ट्रिप:strip एरिया:area जॉब:job फोटो:photo फोन:phone फ़ोन:phone
मोबाइल:mobile कैमरा:camera कैम:cam रेडियो:radio लोकेशन:location आईडिया:idea आइडिया:idea लिटरली:literally स्लो:slow
स्कूल:school कॉलेज:college ऑफिस:office डॉक्टर:doctor पार्टी:party फ्री:free फ्रेंड:friend ब्रेकअप:breakup गर्लफ्रेंड:girlfriend
बॉयफ्रेंड:boyfriend एवरीथिंग:everything वर्किंग:working परफेक्टली:perfectly मेन:main एल:L श:sh सी:si अकेले:akele
वर्क:work वर्ड:word समथिंग:something डाउन:down एक्चुअली:actually एक्सेस:access क्रैश:crash मेक:make डम:dumb लेट:let देन:then
फिश:fish फिशिंग:fishing ड्राइवर:driver ड्राइव:drive पोस्टर:poster पोस्ट:post प्रॉपर:proper फ्रेम:frame लुक्स:looks खान:Khan डॉलर:dollar
ट्राई:try गोइंग:going मीन:mean ओनली:only देयर:there कैरी:carry ड्रेस:dress यूज:use यूज़:use सेफ:safe रिपेयर:repair क्लोज:close ब्लर:blur
टाइमर:timer एनी:any ब्लैक:black सेल:sell न्यू:new गेस:guess अंडर:under पिक:pick बेस:base स्प्रे:spray पेट्रोल:petrol एटीएम:ATM
प्राइवेट:private जीपीएस:GPS सिस्टम:system जंप:jump ग्रोसरी:grocery बॉस:boss इफ:if ऑलमोस्ट:almost ऑप्शन:option नंबर:number स्क्रीन:screen
हेलीकॉप्टर:helicopter शोज़:shows शोज:shows व्हाट्स:what's डाउनलोड:download ऐप:app नेक्स्ट:next पावर:power किल:kill डेड:dead रन:run
फायर:fire हिट:hit मिस:miss स्पेशल:special सीरियस:serious ट्रस्ट:trust प्रॉमिस:promise एंजॉय:enjoy फेवरेट:favourite इम्पॉर्टेंट:important
प्रॉब्लम:problem स्वीट:sweet रीजन:region रिएक्शन:reaction विनर:winner स्कोर:score रिजल्ट:result फैन:fan फैंस:fans ऑडियंस:audience कंटेस्टेंट:contestant
परफॉर्मेंस:performance परफॉर्म:perform टैलेंट:talent फाइनल:final लेसबियन:lesbian टू:to थ्री:three फोर:four फाइव:five सिक्स:six टेन:ten
हंड्रेड:hundred थाउज़ेंड:thousand वाज़:was वर:were देम:them हिम:him हिज:his शी:she दैन:than व्हेन:when व्हिच:which हू:who ऑर:or
नेवर:never ऑलवेज:always बिकॉज़:because बिकॉज:because रियली:really आल्सो:also मेबी:maybe श्योर:sure ग्रेट:great कूल:cool
अमेजिंग:amazing अमेज़िंग:amazing परफेक्ट:perfect वाओ:wow वॉव:wow फिल्म:film
लूसिया:Lucia लुसिया:Lucia ब्रायन:Brian जसन:Jason डकी:Ducky डोनी:Donny हैमलेट:Hamlet
"""

_URDU = """
ہے:hai ہیں:hain ہوں:hoon ہو:ho تھا:tha تھی:thi تھے:the میں:mein نہیں:nahi کیا:kya کیوں:kyun کیسے:kaise کی:ki کا:ka کو:ko کے:ke سے:se اور:aur
یہ:yeh وہ:woh پر:par پہ:pe تو:toh بھی:bhi ہی:hi نے:ne کہ:ke گیا:gaya گئی:gayi گئے:gaye رہا:raha رہی:rahi رہے:rahe ہم:hum تم:tum آپ:aap
مجھے:mujhe میرا:mera میری:meri میرے:mere تیرا:tera بہت:bahut اچھا:accha کچھ:kuch کوئی:koi اب:ab جب:jab تب:tab کب:kab کہاں:kahan
یہاں:yahan وہاں:wahan ایسے:aise ساتھ:saath بعد:baad پہلے:pehle ابھی:abhi پھر:phir لیکن:lekin مگر:magar اگر:agar کر:kar کرو:karo کرنا:karna
ہوا:hua ہوئی:hui دیا:diya لیا:liya بھائی:bhai یار:yaar ٹھیک:theek ایک:ek دو:do سب:sab مطلب:matlab چاہیے:chahiye لیے:liye والا:wala
والی:wali والے:wale اپنا:apna اپنی:apni اپنے:apne پولیس:police مال:maal دل:dil محبت:mohabbat زندگی:zindagi وقت:waqt بات:baat گھر:ghar
کام:kaam پیسے:paise آج:aaj کل:kal دن:din رات:raat لوگ:log دنیا:duniya خدا:khuda عشق:ishq یاد:yaad نام:naam جان:jaan
"""

# "आई" is both "I" and "aayi" (came), "कम" both "kam" (less) and "come": the next word decides
_BEFORE_ON = "ऑन"
_AFTER_ENGLISH_I = set("लव डोंट एम कांट विल थिंक नो हैव वांट डू वाज नीड कैन डिड मीन स्वेयर लाइक गेस गॉट".split())


def _norm(word: str) -> str:
    return unicodedata.normalize("NFC", word.replace("‌", "").replace("‍", ""))


def _table(*blocks: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for block in blocks:
        for pair in block.split():
            word, roman = pair.split(":", 1)
            out.setdefault(_norm(word), roman)
    return out


WORDS = _table(_HINDI, _ENGLISH, _URDU)


def _listed(word: str) -> Optional[str]:
    for key in (word, word.replace("ँ", "ं"), word.replace("़", "")):       # chandrabindu / nukta spellings vary
        if key in WORDS:
            return WORDS[key]
    return None


# --------------------------------------------------------------------------------------------- #
# Spelling rules
# --------------------------------------------------------------------------------------------- #
_CONSONANTS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरऱलळवशषसह",
                       "k kh g gh n ch chh j jh n t th d dh n t th d dh n p f b bh m y r r l l v sh sh s h".split()))
_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y", "न": "n"}
_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
           "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e", "ऒ": "o"}
_MATRAS = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
           "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o"}
_SCHWA = "@"                  # the unwritten "a" of a consonant, which Hindi often does not say


def _devanagari(word: str) -> str:
    units: List[List] = []                  # [consonants, vowel, nasal, is_cluster]
    onset: List[str] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in _CONSONANTS:
            sound, base = _CONSONANTS[ch], ch
            if i + 1 < n and word[i + 1] == "़":
                sound, i = _NUKTA.get(base, sound), i + 1
            onset.append(sound)
            nxt = word[i + 1] if i + 1 < n else ""
            if nxt == "्":             # virama: this consonant joins the next one
                i += 2
                continue
            vowel = _MATRAS.get(nxt)
            if vowel is not None:
                i += 1
            joined = "".join(onset).replace("jn", "gy")
            units.append([joined, vowel if vowel is not None else _SCHWA, False, len(onset) > 1])
            onset = []
        elif ch in _VOWELS:
            glide = ["y"] if ch == "ए" and not onset and units and units[-1][1] in ("i", "ii") else []
            units.append(["".join(onset + glide), _VOWELS[ch], False, len(onset) > 1])
            onset = []
        elif ch in "ंँ" and units:
            units[-1][2] = True
        elif ch == "ः" and units:
            units[-1][1] = ("a" if units[-1][1] == _SCHWA else units[-1][1]) + "h"
        elif ch == "ॐ":
            units.append(["", "om", False, False])
        i += 1
    if onset:                               # the word ends in a virama
        units.append(["".join(onset), "", False, len(onset) > 1])

    last = len(units) - 1
    for k in range(last, 0, -1):            # right to left; the first syllable always keeps its vowel
        consonants, vowel, nasal, cluster = units[k]
        if vowel != _SCHWA or nasal:
            continue
        if k == last:
            if not (cluster and consonants.endswith(("y", "r", "v"))):          # satya, mitra
                units[k][1] = ""
        elif not cluster and units[k - 1][1] and units[k + 1][1] and not units[k + 1][3]:
            units[k][1] = ""

    out = []
    for k, (consonants, vowel, nasal, _cluster) in enumerate(units):
        closed = k == last - 1 and units[last][1] == "" and not units[last][2] and not units[last][3]
        if vowel == _SCHWA:
            vowel = "a"
        elif vowel == "aa":
            long_a = (k == 0 and not consonants) or closed or (nasal and last == 0)
            vowel = "aa" if long_a else "a"
        elif vowel == "ii":
            vowel = "ee" if closed else "i"
        elif vowel == "uu":
            vowel = "oo" if closed else "u"
        out.append(consonants + vowel)
        if nasal:
            following = units[k + 1][0] if k < last else ""
            out.append("m" if following.startswith(("p", "b", "m")) else "n")
    return "".join(out)


_ARABIC = {"ا": "a", "آ": "aa", "أ": "a", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh",
           "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t",
           "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n",
           "و": "o", "ؤ": "o", "ہ": "h", "ه": "h", "ۂ": "h", "ة": "t", "ھ": "h", "ی": "i", "ي": "i", "ى": "a", "ئ": "i", "ے": "e",
           "ۓ": "e", "ء": "", "َ": "a", "ِ": "i", "ُ": "u", "ً": "an"}


def _urdu(word: str) -> str:
    word = unicodedata.normalize("NFC", word)
    out: List[str] = []
    for k, ch in enumerate(word):
        sound = _ARABIC.get(ch, "")
        if ch == "و" and k == 0:
            sound = "w"
        elif ch in "یي" and k == 0:
            sound = "y"
        elif ch in "ہه" and k == len(word) - 1 and k > 0:
            sound = "a"
        elif ch == "ّ" and out:                 # shadda doubles the consonant
            sound = out[-1]
        if sound:
            out.append(sound)
    # Urdu does not write short vowels: give an opening pair of consonants one so it can be read
    if len(out) > 1 and out[0][0] not in "aeiou" and out[1][0] not in "aeiouhy":
        out.insert(1, "a")
    return "".join(out)


def _by_rules(word: str) -> str:
    return _devanagari(word) if "ऀ" <= word[0] <= "ॿ" else _urdu(word)


# --------------------------------------------------------------------------------------------- #
# Remembered AI answers
# --------------------------------------------------------------------------------------------- #
_lock = threading.Lock()
_memory: Dict[str, Dict[str, str]] = {}          # cache file -> {word: roman}


def _cache_file() -> Path:
    return Path(config.LOCAL_OUTPUT_DIR) / "cache" / "roman_words.json"


def _learned() -> Dict[str, str]:
    path = _cache_file()
    with _lock:
        words = _memory.get(str(path))
        if words is None:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                words = {str(k): str(v) for k, v in data.items()}
            except (OSError, ValueError, AttributeError):
                words = {}
            _memory[str(path)] = words
        return words


def _remember(new: Dict[str, str]) -> None:
    words = _learned()
    path = _cache_file()
    with _lock:
        words.update(new)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass


PROMPT = """Write each Hindi or Urdu word below in Roman (Latin) letters, the way people type it in chat (Hinglish / Roman Urdu).
Do NOT translate: नहीं is "nahi", क्यों is "kyun", زندگی is "zindagi". When a word is really an English word written in Hindi or
Urdu letters, give its normal English spelling: वीडियो is "video".
Respond ONLY with valid JSON (no markdown): {{"words":{{"1":"...","2":"..."}}}} with every number below.
Words: {words}"""

_ROMAN_ANSWER = re.compile(r"^[A-Za-z0-9'’.\-]+( [A-Za-z0-9'’.\-]+){0,2}$")


def prepare(texts: Iterable[str], llm_fn: Optional[Callable[[str], str]] = None) -> int:
    """Ask the AI, once and within a time limit, about words that neither the list nor the cache knows.

    Returns how many words it answered. Never raises (except when the job is cancelled): the rules cover the rest.
    """
    if llm_fn is None:
        return 0
    learned = _learned()
    unknown: Dict[str, None] = {}
    for text in texts:
        for run in _RUN.findall(str(text or "")):
            word = _norm(run)
            if word and word != "आई" and _listed(word) is None and word not in learned and _ANY.search(word):
                unknown[word] = None
    if not unknown:
        return 0
    from ..highlights import _parse_json_loose
    from . import control
    from .llm import time_budget

    todo, answered = list(unknown), 0
    try:
        with time_budget(LLM_SECONDS):
            for at in range(0, len(todo), CHUNK_WORDS):
                chunk = todo[at:at + CHUNK_WORDS]
                numbered = {str(i + 1): w for i, w in enumerate(chunk)}
                parsed = _parse_json_loose(llm_fn(PROMPT.format(words=json.dumps(numbered, ensure_ascii=False))))
                got = parsed.get("words") if isinstance(parsed, dict) else None
                good: Dict[str, str] = {}
                for key, value in (got.items() if isinstance(got, dict) else []):
                    value = " ".join(str(value).split())
                    if key in numbered and len(value) <= 40 and _ROMAN_ANSWER.match(value):
                        good[numbered[key]] = value.rstrip(".")
                if good:
                    _remember(good)
                    answered += len(good)
    except control.Cancelled:
        raise
    except Exception as e:  # noqa: BLE001 - spelling rules take over
        print(f"[roman] the AI could not spell {len(todo) - answered} word(s) ({str(e)[:140]}); using spelling rules", flush=True)
    return answered


# --------------------------------------------------------------------------------------------- #
# Using it
# --------------------------------------------------------------------------------------------- #
def needed(text: str) -> bool:
    return bool(_ANY.search(str(text or "")))


def _token(token: str, following: str, learned: Dict[str, str]) -> str:
    def swap(match) -> str:
        word = _norm(match.group(0))
        if not word:
            return ""
        after = _norm("".join(_RUN.findall(following)))
        if word == "आई":
            return "I" if after in _AFTER_ENGLISH_I else "aayi"
        if word == "कम" and after == _norm(_BEFORE_ON):
            return "come"
        return _listed(word) or learned.get(word) or _by_rules(word)

    token = token.translate(_DIGITS)
    for mark, plain in _PUNCT.items():
        token = token.replace(mark, plain)
    return _RUN.sub(swap, token)


def _tokens(tokens: Sequence[str]) -> List[str]:
    learned = _learned()
    return [_token(t, tokens[i + 1] if i + 1 < len(tokens) else "", learned) if needed(t) else t for i, t in enumerate(tokens)]


def text(value: str) -> str:
    """A whole line in Roman letters. Text without Hindi or Urdu letters comes back unchanged."""
    value = str(value or "")
    if not needed(value):
        return value
    return "\n".join(" ".join(t for t in _tokens(line.split()) if t) for line in value.split("\n"))


def _keep_sound_notes(tokens: List[str]) -> List[str]:
    """Romanize the spoken words only: "[हंसी]" style notes stay as they are, the editor reads them as reactions."""
    roman = _tokens(tokens)
    first_close = next((i for i, t in enumerate(tokens) if re.search(r"[\])]", t)), None)
    first_open = next((i for i, t in enumerate(tokens) if re.search(r"[\[(]", t)), None)
    start, closer, out = 0, "", []
    if first_close is not None and (first_open is None or first_close < first_open):      # a note begun on the previous line
        start = first_close + 1
        out = list(tokens[:start])
    for original, new in zip(tokens[start:], roman[start:]):
        bare = original.strip()
        if not closer and bare.startswith(("[", "(")):
            closer = "]" if bare.startswith("[") else ")"
        out.append(original if closer else new)
        if closer and closer in bare:
            closer = ""
    return out


def segments(items: Sequence[Dict], spans: Sequence[Tuple[float, float]] = ()) -> List[Dict]:
    """A copy of transcript segments with Roman words (only those touching ``spans``, when given). Timings are the same."""
    out: List[Dict] = []
    for seg in items or []:
        if spans and not any(seg["end"] > a and seg["start"] < b for a, b in spans):
            out.append(seg)
            continue
        words = seg.get("words")
        if words:
            roman = _keep_sound_notes([str(w["text"]) for w in words])
            out.append({**seg, "words": [{**w, "text": t} for w, t in zip(words, roman)], "text": " ".join(t.strip() for t in roman)})
        else:
            out.append({**seg, "text": " ".join(_keep_sound_notes(str(seg.get("text", "")).split()))})
    return out


def hints(notes: Dict) -> Dict:
    """The AI's edit plan ({hook, popups, keywords}) in Roman letters."""
    return {"hook": text(notes.get("hook")),
            "popups": [{**p, "text": text(p.get("text"))} for p in notes.get("popups") or []],
            "keywords": [text(k) for k in notes.get("keywords") or []],
            **({"memes": notes["memes"]} if "memes" in notes else {})}
