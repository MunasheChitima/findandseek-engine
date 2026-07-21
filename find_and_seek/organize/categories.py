"""The document-category taxonomy — the single source of truth for "where does
this belong?".

This replaces the scattered, free-form `document_type` guessing (two different
prompts, an ignorable enum, post-hoc canonicalisation) with ONE deliberate set
of categories, each with a real definition the classifier reasons against. A
category is a considered placement, not a label the model happened to emit.

Design contract:
  * Every category has a `definition` AND `not_` (what it excludes) — the model
    can only place a document correctly if it knows the boundaries (an invoice is
    a bill with an amount due, NOT a store page that shows prices).
  * The taxonomy is USER-EXTENSIBLE: `load_taxonomy()` returns the built-in
    categories plus any the user has added (stored in the DB). Adding a category
    is meant to make the app *re-engage* — re-classify against the expanded set —
    not merely paint a new label (see organize/reclassify.py).
  * `NEEDS_REVIEW` is not a category the model picks; it's where the classifier
    routes a document when it is not confident or the text is too thin to judge.
    Honest uncertainty beats a confident mistake.
"""

from __future__ import annotations

from dataclasses import dataclass

NEEDS_REVIEW = "needs-review"


@dataclass(frozen=True)
class Category:
    slug: str
    label: str
    definition: str
    not_: str = ""          # what this category explicitly EXCLUDES
    builtin: bool = True


# Built-in taxonomy. Slugs are stable identifiers; labels are what the UI shows.
BUILTIN_CATEGORIES: tuple[Category, ...] = (
    Category("invoice", "Invoice",
             "A bill that requests or confirms a specific payment — it has an amount due, an invoice or account number, or a 'bill to'.",
             not_="a product/price list, a store or shopping page, or a general payment record"),
    Category("receipt", "Receipt",
             "Proof of a completed purchase or payment: itemised goods/services with a total paid and a date.",
             not_="a bill still owing (that is an invoice), a pay slip (that is payslip), or a product catalogue"),
    Category("payslip", "Payslip",
             "A pay slip or salary statement from an employer: gross/net pay for a pay period, "
             "payment date, tax withheld, year-to-date totals for an employee.",
             not_="an invoice, a purchase receipt, or a timesheet to be filled in"),
    Category("contract", "Contract",
             "An agreement setting out terms and obligations between parties — clauses, signatures, effective/termination dates.",
             not_="a court filing or witness statement (that is legal)"),
    Category("report", "Report",
             "Analysis, findings, status, a review, or a plan — e.g. an operating review, market analysis, or assessment plan.",
             not_="reference/instructional material such as a manual or curriculum"),
    Category("letter", "Letter",
             "A piece of correspondence written TO a named recipient, with a salutation "
             "(e.g. 'Dear …') and a sign-off (e.g. 'Yours sincerely', 'Regards') — a cover "
             "letter, formal letter, or notice letter.",
             not_="an email or instant-message/chat thread (LinkedIn message, chat export), "
                  "a policy or code of conduct, a contract or agreement, or any document "
                  "that merely names or mentions a recipient without being a letter"),
    Category("email", "Email",
             "An email message or thread, with From/To/Subject or quoted reply history.",
             not_="a posted letter, or an instant-message/chat conversation"),
    Category("chat", "Chat / Transcript",
             "A chat or instant-message conversation, or an AI-assistant transcript — an "
             "exported message thread with back-and-forth turns or timestamps (a LinkedIn / "
             "WhatsApp / Slack / iMessage export, or a ChatGPT / Gemini / Claude conversation). "
             "Includes conversations whose speaker labels were stripped by the export: "
             "alternating first-person questions and second-person answers or advice still "
             "make it a chat.",
             not_="an email with From/To/Subject headers, or a letter written to a named "
                  "recipient with a salutation and sign-off"),
    Category("cv", "CV / Résumé",
             "A person's curriculum vitae or résumé: their experience, skills, and employment history.",
             not_="a job posting or an application form"),
    Category("spreadsheet", "Spreadsheet / Data",
             "Tabular data — a spreadsheet or CSV of rows and columns: a catalogue, list, inventory, or records export.",
             not_="a single bill (invoice) even if it contains figures, or the text dump of a web app or dashboard (that is a web page)"),
    Category("web-page", "Web page / Listing",
             "A saved web page or online listing — a store/product page, search results, or browser-rendered site (navigation bars, 'Add to Cart', 'Results', a retailer like Amazon or DoorDash).",
             not_="an actual invoice or receipt issued to you"),
    Category("presentation", "Presentation",
             "A slide deck or presentation — titled slides, speaker points.",
             not_="a written report or document"),
    Category("form", "Form / Certificate",
             "An official form, certificate, application, or template — issued or to be filled in (e.g. a separation certificate, application form).",
             not_="a contract between parties, or a store docket / proof of purchase (that is a receipt) even when it carries codes and boxes"),
    Category("reference", "Reference / Guide",
             "Reference or instructional material — a manual, specification, guide, study design, curriculum, syllabus, or FAQ.",
             not_="a one-off analysis or status report"),
    Category("medical", "Medical record",
             "A medical or health record — a prescription, dispensing record, claim, lab result, or clinical note.",
             not_="a general invoice that merely happens to be from a clinic"),
    Category("legal", "Legal document",
             "A legal document other than a contract — a court filing, witness statement, judgment, affidavit, or formal legal notice.",
             not_="a contract or agreement"),
    Category("note", "Note",
             "A personal note, memo, or short informal jotting.",
             not_="a formal report or letter"),
    Category("other", "Other",
             "A real document that genuinely does not fit any other category.",
             not_="anything that fits a category above"),
)

BUILTIN_SLUGS: frozenset[str] = frozenset(c.slug for c in BUILTIN_CATEGORIES)


def load_taxonomy(conn=None) -> list[Category]:
    """The active taxonomy: built-ins plus any user-added categories (DB-backed).

    `conn` is an optional sqlite connection; when given, user categories from the
    `categories` table are appended. The classifier always classifies against the
    *current* taxonomy, so a user addition is honoured the next time a document is
    classified — that's what makes adding a category re-engage the app rather than
    just relabel.
    """
    cats = list(BUILTIN_CATEGORIES)
    if conn is not None:
        try:
            rows = conn.execute(
                "SELECT slug, label, definition, COALESCE(excludes,'') AS excludes "
                "FROM categories WHERE builtin=0 ORDER BY label"
            ).fetchall()
            have = {c.slug for c in cats}
            for r in rows:
                slug = r["slug"] if not isinstance(r, tuple) else r[0]
                if slug in have:
                    continue
                label = r["label"] if not isinstance(r, tuple) else r[1]
                definition = r["definition"] if not isinstance(r, tuple) else r[2]
                excludes = r["excludes"] if not isinstance(r, tuple) else r[3]
                # Keep 'other' last so user categories are offered before the catch-all.
                cats.insert(len(cats) - 1, Category(slug, label, definition, excludes, builtin=False))
        except Exception:  # noqa: BLE001 — table may not exist yet (pre-migration)
            pass
    return cats


def category_label(slug: str, conn=None) -> str:
    """Human label for a slug; falls back to a title-cased slug."""
    if slug == NEEDS_REVIEW:
        return "Needs review"
    for c in load_taxonomy(conn):
        if c.slug == slug:
            return c.label
    return slug.replace("-", " ").title()
