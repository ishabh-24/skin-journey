/**
 * Generates a combined OTC-style routine from acne severity, eczema pattern, and
 * highest-inflammation facial zone. Not medical advice — educational only.
 */

import type { EczemaBucket, RegionScores, SeverityBucket } from '../types/models';

export type RoutineProduct = {
  name: string;
  note: string;
};

export type RoutineStep = {
  order: number;
  title: string;
  howToUse: string;
  products: RoutineProduct[];
};

export type CombinedSkincareRoutine = {
  headline: string;
  intro: string;
  zoneLine: string;
  morning: RoutineStep[];
  night: RoutineStep[];
  cautions: string[];
};

export type RoutinePeriod = 'Morning' | 'Night';

export type WizardRoutineStep = {
  id: string;
  period: RoutinePeriod;
  order: number;
  title: string;
  howToUse: string;
  options: RoutineProduct[];
};

const WIZARD_OPTION_FILLERS: RoutineProduct[] = [
  {
    name: 'Ask a pharmacist / clinician',
    note: 'Pick a product in the same category and similar strength as the suggestions above.',
  },
  {
    name: 'Drugstore equivalent',
    note: 'Match the active ingredient class on the label (e.g. BP %, BHA, ceramides, mineral SPF).',
  },
  {
    name: 'Trusted brand swap',
    note: 'Same step from another line your skin has tolerated before.',
  },
];

/** Ensures exactly three tappable options per wizard step (pads with placeholders if needed). */
export function normalizeRoutineProductOptions(products: RoutineProduct[]): RoutineProduct[] {
  const used = products.slice(0, 3);
  const out = [...used];
  let f = 0;
  while (out.length < 3 && f < WIZARD_OPTION_FILLERS.length) {
    out.push(WIZARD_OPTION_FILLERS[f]);
    f += 1;
  }
  while (out.length < 3) {
    out.push({
      name: 'Same-step alternative',
      note: 'Choose any OTC product that fills this step’s role for your skin.',
    });
  }
  return out.slice(0, 3);
}

export function buildWizardSteps(routine: CombinedSkincareRoutine): WizardRoutineStep[] {
  const out: WizardRoutineStep[] = [];
  routine.morning.forEach((s, i) => {
    out.push({
      id: `morning-${i}`,
      period: 'Morning',
      order: s.order,
      title: s.title,
      howToUse: s.howToUse,
      options: normalizeRoutineProductOptions(s.products),
    });
  });
  routine.night.forEach((s, i) => {
    out.push({
      id: `night-${i}`,
      period: 'Night',
      order: s.order,
      title: s.title,
      howToUse: s.howToUse,
      options: normalizeRoutineProductOptions(s.products),
    });
  });
  return out;
}

const ZONE_PRETTY: Record<string, string> = {
  forehead: 'forehead & hairline',
  left_cheek: 'left cheek',
  right_cheek: 'right cheek',
  jawline: 'jawline & chin',
};

function zonePretty(primaryZone: string): string {
  return ZONE_PRETTY[primaryZone] ?? primaryZone.replace(/_/g, ' ');
}

function regionIntensitySummary(rs: RegionScores, top = 3): string {
  const parts = Object.entries(rs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, top)
    .map(([k, v]) => `${zonePretty(k)} ${Math.round(v * 100)}%`);
  return parts.join(' · ');
}

function products(...items: [string, string][]): RoutineProduct[] {
  return items.map(([name, note]) => ({ name, note }));
}

/** Stronger acne actives OK when eczema is absent or mild only. */
function acneActivesLevel(acne: SeverityBucket, eczema: EczemaBucket): 'low' | 'medium' | 'high' {
  if (eczema === 'severe_eczema') return 'low';
  if (eczema === 'mild_eczema') return acne === 'severe' ? 'medium' : 'low';
  if (acne === 'severe') return 'high';
  if (acne === 'moderate') return 'medium';
  return 'low';
}

export function buildCombinedSkincareRoutine(opts: {
  severityBucket: SeverityBucket;
  severityScore: number;
  eczemaBucket: EczemaBucket;
  eczemaLikelihood: number;
  regionScores: RegionScores;
  primaryZone: string;
}): CombinedSkincareRoutine {
  const { severityBucket, severityScore, eczemaBucket, eczemaLikelihood, primaryZone, regionScores } = opts;
  const zone = zonePretty(primaryZone);
  const actives = acneActivesLevel(severityBucket, eczemaBucket);

  const cautions: string[] = [
    'This routine is educational only and not a diagnosis. Stop products that sting, burn, or worsen redness; see a clinician for prescription care if needed.',
    'Introduce one new active at a time (wait ~10–14 days) to see tolerance.',
  ];
  if (eczemaBucket !== 'none') {
    cautions.push('Eczema-prone skin: prioritize barrier repair; avoid starting multiple harsh actives at once.');
  }
  if (severityBucket === 'severe' || eczemaBucket === 'severe_eczema') {
    cautions.push('Higher severity on this scan may warrant in-person dermatology — especially with pain, scarring, or rapidly worsening rash.');
  }

  const introParts = [
    `Built for your latest scan: acne ${severityBucket} (score ${severityScore.toFixed(1)}/10) and eczema-type pattern ${eczemaBucket.replace(/_/g, ' ')} (likelihood ${eczemaLikelihood.toFixed(1)}/10).`,
    `Your model highlights ${zone} as the strongest inflammation signal — spend a few extra seconds blending treatments there, but keep actives off eyelids and lips.`,
  ];
  const intro = introParts.join(' ');

  const zoneLine = `Focus zone: ${zone} — apply treatments evenly there after patch-testing new products behind the ear for 48h. Relative inflammation from this scan: ${regionIntensitySummary(regionScores)}.`;

  // --- Morning ---
  const morning: RoutineStep[] = [];
  let o = 1;

  morning.push({
    order: o++,
    title: 'Cleanse (gentle, lukewarm water)',
    howToUse:
      'AM: 30–60s massage, rinse thoroughly. Pat dry — do not rub. If you wore overnight occlusive, one light pass is enough.',
    products: products(
      ['CeraVe Hydrating Cleanser', 'Non-foaming; good with retinoids or dryness.'],
      ['La Roche-Posay Toleriane Hydrating Gentle Cleanser', 'Fragrance-sensitive option.'],
      ['Vanicream Gentle Facial Cleanser', 'Minimal ingredients; eczema-friendly baseline.'],
    ),
  });

  if (actives === 'high' && eczemaBucket === 'none') {
    morning.push({
      order: o++,
      title: 'Benzoyl peroxide (short contact or wash-off)',
      howToUse:
        'Use 2.5–5% BP as a thin layer for 1–3 minutes then rinse, OR a BP wash in shower 1× daily to start. Increase contact time weekly if tolerated. Rinse well before moisturizer + SPF.',
      products: products(
        ['PanOxyl Acne Creamy Wash 4%', 'Wash-off format reduces staining vs leave-on.'],
        ['Neutrogena Stubborn Acne AM Treatment 2.5%', 'Lower % often less irritating.'],
        ['CeraVe Acne Foaming Cream Cleanser 4% BP', 'Combine cleanse + BP in one step if preferred.'],
      ),
    });
  } else if (actives === 'medium' && eczemaBucket !== 'severe_eczema') {
    morning.push({
      order: o++,
      title: 'Salicylic acid (low %, 2–4×/week to start)',
      howToUse:
        'Apply to oily/acne-prone zones after pat-dry. Skip this step on days skin feels tight. Always follow with moisturizer + SPF.',
      products: products(
        ['Paula’s Choice 2% BHA Liquid', 'Classic leave-on BHA; use sparingly on cheeks if sensitive.'],
        ['CeraVe SA Cleanser', 'Rinse-off option — gentler entry to salicylic acid.'],
        ['The Inkey List Salicylic Acid Cleanser', 'Budget rinse-off BHA.'],
      ),
    });
  } else if (eczemaBucket !== 'none') {
    morning.push({
      order: o++,
      title: 'Barrier-soothing layer (optional serum or essence)',
      howToUse:
        'Before moisturizer: thin layer of hyaluronic acid or niacinamide on damp skin if tolerated. Skip acids this AM if skin is flaky or stinging.',
      products: products(
        ['La Roche-Posay Cicaplast B5 Gel', 'Soothing; pairs with irritated areas.'],
        ['Vichy Mineral 89', 'Hyaluronic + mineral water base.'],
        ['The Ordinary Niacinamide 10% + Zinc 1%', 'Oil control; patch-test if eczema-prone.'],
      ),
    });
  }

  morning.push({
    order: o++,
    title: 'Moisturizer (cream or lotion)',
    howToUse:
      'While skin is slightly damp, apply a generous layer. For eczema tendency, favor creams/ointments over gels. Wait 3–5 min before SPF.',
    products: products(
      ['CeraVe Moisturizing Cream', 'Ceramides + petrolatum barrier; widely tolerated.'],
      ['Eucerin Advanced Repair Cream', 'Urea option for very rough texture (avoid broken skin).'],
      ['Aveeno Eczema Therapy Daily Moisturizing Cream', 'Colloidal oatmeal; eczema-prone friendly.'],
    ),
  });

  morning.push({
    order: o++,
    title: 'Sunscreen SPF 30+ (broad spectrum)',
    howToUse:
      'Two finger-lengths for face + neck. Reapply every 2h if outdoors/sweating. Chemical vs mineral is personal — mineral can be whiter but gentler for some reactive skin.',
    products: products(
      ['EltaMD UV Clear SPF 46', 'Popular under makeup; niacinamide.'],
      ['La Roche-Posay Anthelios Melt-In Milk SPF 60', 'High UVA protection.'],
      ['CeraVe Hydrating Mineral Sunscreen SPF 30', 'Mineral filter; better for some sensitive types.'],
    ),
  });

  // --- Night ---
  const night: RoutineStep[] = [];
  o = 1;

  night.push({
    order: o++,
    title: 'First cleanse (if sunscreen/makeup) then gentle cleanser',
    howToUse:
      'PM: oil or balm cleanser to dissolve SPF, then your gentle non-stripping cleanser. Lukewarm water only.',
    products: products(
      ['Vanicream Gentle Facial Cleanser', 'Second cleanse / single cleanse option.'],
      ['CeraVe Hydrating Cleanser', 'Repeat from AM if you prefer one bottle.'],
      ['The Inkey List Oat Cleansing Balm', 'First step to melt sunscreen — rinse then follow with gel/cream cleanser.'],
    ),
  });

  if (eczemaBucket === 'severe_eczema') {
    night.push({
      order: o++,
      title: 'Barrier repair (priority over acne acids)',
      howToUse:
        'Apply a thick bland emollient to face and any dry patches. You can “slug” with petrolatum on top 2–3×/week if not acne-clog prone — skip slugging on very oily T-zone if you break out.',
      products: products(
        ['Vaseline Original Healing Jelly', 'Occlusive seal over moisturizer on rough nights.'],
        ['Aquaphor Healing Ointment', 'Lanolin blend; heavier than cream.'],
        ['CeraVe Healing Ointment', 'Ceramide + petrolatum hybrid.'],
      ),
    });
  }

  if (actives !== 'low' && eczemaBucket !== 'severe_eczema') {
    night.push({
      order: o++,
      title: 'Adapalene 0.1% (retinoid) — acne backbone',
      howToUse:
        'Start 2 nights/week, pea-sized for whole face (avoid corners of nose/mouth/eyes). Increase to nightly over 4–8 weeks if no irritation. Buffer: moisturizer → adapalene → moisturizer (“sandwich”) if needed.',
      products: products(
        ['Differin Gel 0.1% Adapalene', 'OTC standard; use at night only.'],
        ['La Roche-Posay Effaclar Adapalene Gel 0.1%', 'Alternative brand adapalene where available.'],
        ['AcneFree Adapalene Gel 0.1%', 'Store-brand adapalene option in many regions.'],
      ),
    });
  } else if (actives === 'low' && severityBucket !== 'mild') {
    night.push({
      order: o++,
      title: 'Low-strength retinoid or retinal (optional, 1–2×/week)',
      howToUse:
        'Because eczema risk or sensitivity is elevated, use a pea-sized amount infrequently OR skip until barrier feels calm for 1 week straight.',
      products: products(
        ['Differin Gel 0.1%', 'Still the default OTC retinoid if you patch-test OK.'],
        ['CeraVe Resurfacing Retinol Serum', 'Lower irritation potential than adapalene for some.'],
        ['The Ordinary Granactive Retinoid 2% in Squalane', 'Milder retinoid ester; patch-test.'],
      ),
    });
  }

  if (actives === 'high' && eczemaBucket === 'none') {
    night.push({
      order: o++,
      title: 'Benzoyl peroxide gel (leave-on) OR alternate nights with adapalene',
      howToUse:
        'If not using adapalene this night: thin BP layer on breakout-prone areas. If combining, use BP AM and adapalene PM, or alternate nights — rarely same night unless dermatologist-guided.',
      products: products(
        ['Neutrogena On-the-Spot 2.5% BP', 'Spot or thin layer.'],
        ['La Roche-Posay Effaclar Duo (+)', 'BP + LHA in some markets — read label for BP %.'],
        ['Humane Acne Treatment 2.5% BP', 'Simple gel formula.'],
      ),
    });
  }

  if (severityBucket === 'mild' && eczemaBucket === 'none') {
    night.push({
      order: o++,
      title: 'Azelaic acid 10% (optional, 3–4×/week)',
      howToUse:
        'Thin layer after moisturizer dries or before moisturizer if tolerated. Helps tone and mild breakouts with usually lower irritation than retinoids.',
      products: products(
        ['The Ordinary Azelaic Acid Suspension 10%', 'Cream-gel texture; can pill under some SPF — pat, don’t rub.'],
        ['Paula’s Choice 10% Azelaic Acid Booster', 'Silicone base; often layers more easily.'],
        ['Geek & Gorgeous A-Game 10 Azelaic Serum', 'Azelaic in serum form where available.'],
      ),
    });
  }

  if (eczemaBucket === 'mild_eczema') {
    night.push({
      order: o++,
      title: 'Anti-itch / colloidal oatmeal (as needed)',
      howToUse:
        'On itchy patches only, after moisturizer: a small amount of OTC hydrocortisone 1% cream for up to 7 days on face only if clinician-approved; otherwise stick to oatmeal creams.',
      products: products(
        ['Aveeno Eczema Therapy Itch Relief Balm', 'Colloidal oatmeal; no steroid.'],
        ['Eucerin Eczema Relief Cream', 'Licochalcone A soothing formula.'],
        ['Vanicream HC 1% Hydrocortisone Anti-Itch Cream', 'Short-term only — confirm with pharmacist/clinician for face use.'],
      ),
    });
  }

  night.push({
    order: o++,
    title: 'Night moisturizer (equal or richer than AM)',
    howToUse:
      'Generous layer after actives dry (wait ~20 min after retinoid if skin is sensitive). Reapply a second layer on dry zones.',
    products: products(
      ['CeraVe Moisturizing Cream', 'Repeat from AM — consistency helps barrier.'],
      ['La Roche-Posay Lipikar AP+ Balm', 'Rich body/face balm; very dry skin.'],
      ['First Aid Beauty Ultra Repair Cream', 'Colloidal oatmeal + shea; patch-test if sensitive to eucalyptus.'],
    ),
  });

  const headline =
    eczemaBucket === 'severe_eczema'
      ? 'Barrier-first routine (acne actives minimized)'
      : severityBucket === 'severe' && eczemaBucket === 'none'
        ? 'Acne-focused routine with strict SPF'
        : 'Balanced routine for acne + skin barrier';

  return {
    headline,
    intro,
    zoneLine,
    morning,
    night,
    cautions,
  };
}
