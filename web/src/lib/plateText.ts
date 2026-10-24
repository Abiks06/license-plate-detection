/**
 * Turns the plate reader's per-slot probabilities into text, using Ghana's plate formats.
 * Same algorithm as training/scripts/plate_text.py (which the accuracy figures come from);
 * keep the two in sync.
 *
 * The reader predicts each of its 10 slots independently, so it can say "AS4633I2" although a
 * Ghanaian plate never has a letter in its year. Ghanaian plates are two letters, 1-4 digits,
 * then a two-digit year or one letter (GR 3966-21, CD 590 18, GR 672 H, DV 334J). `decodePlate`
 * picks the most probable string of that shape, unless the free reading is far more probable
 * (a foreign plate), in which case it keeps that.
 */

export const ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_';
const PAD = ALPHABET.indexOf('_');
const DIGITS = [...Array(10).keys()];
const LETTERS = [...Array(26).keys()].map((i) => i + 10);
/** How much less likely (in nats) the Ghanaian reading may be and still win: ~400x. */
const MARGIN = 6;

export type PlateChar = { char: string; prob: number };
export type PlateRead = {
  /** Letters and digits only, e.g. "GR396621". */
  text: string;
  /** As printed, e.g. "GR 3966-21". */
  plate: string;
  chars: PlateChar[];
  /** Product of the character probabilities. */
  confidence: number;
  format: 'ghana' | 'free';
};

type Cls = 'L' | 'D';

function* patterns(slots: number): Generator<Cls[]> {
  for (let n = 1; n <= 4; n++) {
    for (const suffix of [['D', 'D'], ['L']] as Cls[][]) {
      const p: Cls[] = ['L', 'L', ...Array<Cls>(n).fill('D'), ...suffix];
      if (p.length <= slots) yield p;
    }
  }
}

const ln = (p: number) => Math.log(Math.max(p, 1e-12));

/** probs: row-major [slots][37]. */
export function decodePlate(probs: Float32Array, slots: number, ghana = true): PlateRead {
  const V = ALPHABET.length;
  const at = (i: number, c: number) => probs[i * V + c];
  const argmax = (i: number, ids: number[]) => ids.reduce((b, c) => (at(i, c) > at(i, b) ? c : b), ids[0]);

  const all = [...Array(V).keys()];
  const free = [...Array(slots).keys()].map((i) => argmax(i, all));
  const freeScore = free.reduce((s, c, i) => s + ln(at(i, c)), 0);

  let best: { score: number; pattern: Cls[]; ids: number[] } | null = null;
  if (ghana) {
    for (const pattern of patterns(slots)) {
      const ids = pattern.map((cls, i) => argmax(i, cls === 'D' ? DIGITS : LETTERS));
      let score = ids.reduce((s, c, i) => s + ln(at(i, c)), 0);
      for (let i = pattern.length; i < slots; i++) score += ln(at(i, PAD));
      if (!best || score > best.score) best = { score, pattern, ids };
    }
  }

  const useGhana = best !== null && best.score >= freeScore - MARGIN;
  const chosen = useGhana ? best!.ids.map((c, i) => [i, c]) : free.map((c, i) => [i, c]).filter(([, c]) => c !== PAD);
  const chars = chosen.map(([i, c]) => ({ char: ALPHABET[c], prob: at(i, c) }));
  const text = chars.map((c) => c.char).join('');
  return {
    text,
    plate: useGhana ? display(text, best!.pattern) : text,
    chars,
    confidence: chars.reduce((p, c) => p * c.prob, chars.length ? 1 : 0),
    format: useGhana ? 'ghana' : 'free',
  };
}

/** GR396621 -> "GR 3966-21"; GR672H -> "GR 672 H". */
function display(text: string, pattern: Cls[]) {
  return pattern.at(-1) === 'D' ? `${text.slice(0, 2)} ${text.slice(2, -2)}-${text.slice(-2)}` : `${text.slice(0, 2)} ${text.slice(2, -1)} ${text.slice(-1)}`;
}

/**
 * Where each character of `text` sits in `plate`, so the UI can mark doubtful characters in the
 * formatted plate ("GR 3966-21" has spaces and dashes the probabilities don't cover).
 */
export function charSlots(plate: string): (number | null)[] {
  let k = 0;
  return [...plate].map((ch) => (/[A-Z0-9]/.test(ch) ? k++ : null));
}
