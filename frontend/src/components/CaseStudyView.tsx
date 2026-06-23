import styles from './StaticPage.module.css';

/** Static content, rendered verbatim per the project brief. */
export function CaseStudyView() {
  return (
    <div className={styles.page}>
      <h1 className={styles.title}>Case Study: Explainable Fraud Detector</h1>
      <p className={styles.lead}>
        The interesting part of this project wasn't building a fraud classifier. It was deciding what to do after
        choosing a classifier optimized for catching fraud at the expense of generating many false alarms.
      </p>

      <p className={styles.paragraph}>
        <strong>The model decision</strong> I evaluated several model variants and found that XGBoost with SMOTE
        achieved the best F1 score (0.75). However, I chose to deploy XGBoost with class weights instead, achieving
        0.93 recall and 0.34 precision. In a fraud and compliance context, missing real fraud is often more costly
        than investigating a legitimate transaction.
      </p>

      <p className={styles.paragraph}>
        <strong>The tradeoff that followed</strong> That choice created the central design challenge for the rest
        of the system. At 0.34 precision, most transactions flagged as fraud are actually legitimate. Analysts
        would spend much of their time reviewing noise unless the model could explain its decisions efficiently and
        consistently. The retrieval and agent layers exist because of that tradeoff. The deployed model generates
        many false positives, creating a need for scalable review and auditability.
      </p>

      <p className={styles.paragraph}>
        <strong>Grounded retrieval</strong> To address this, I built retrieval grounded in the model's own
        evidence. Rather than using a generic prompt, the transaction's top SHAP factors become the search query
        against a fraud-policy knowledge base. The retrieved policy guidance is then used to generate a
        plain-English explanation of why the transaction was flagged.
      </p>

      <p className={styles.paragraph}>
        <strong>The agent layer</strong> On top of that, I built an agent that reviews the merchant's historical
        activity, retrieves the relevant policy, and recommends whether the case should be escalated for human
        review or cleared. Every recommendation is logged with its supporting evidence and reasoning to create an
        auditable decision trail.
      </p>

      <p className={styles.paragraph}>
        <strong>What the agent surfaced, unplanned</strong> One of the more interesting outcomes emerged from a
        fairness check I ran during development. After identifying a potential relationship involving city
        population, I documented the finding and added a policy note recommending governance review if the effect
        became more pronounced over time. During a later test, the agent independently retrieved and cited that
        note while making a decision. That was the first point where the system stopped feeling like a collection
        of components and started behaving more like an organization that could remember and apply its own prior
        decisions.
      </p>

      <p className={styles.paragraph}>
        <strong>Data quality fixes</strong> Along the way, I also identified and fixed two practical data-quality
        issues. Merchant names contained a documented "fraud_" prefix artifact across the dataset, which I removed
        during preprocessing. I also replaced a distance feature that was based on raw coordinate differences with
        an automatically computed geographic distance, eliminating the need for users to provide information they
        could not reasonably know.
      </p>

      <p className={styles.paragraph}>
        <strong>What this demonstrates</strong> The final system isn't a fraud classifier with explainability,
        retrieval, and agents bolted on. It's a workflow built around a deliberate operational tradeoff: maximize
        fraud detection, accept a higher volume of false positives, and provide reviewers with enough context and
        evidence to handle that noise efficiently, consistently, and transparently.
      </p>
    </div>
  );
}
