import styles from './StaticPage.module.css';

/** Static content, rendered verbatim per the project brief. */
export function AboutView() {
  return (
    <div className={styles.page}>
      <h1 className={styles.title}>About This Project</h1>
      <p className={styles.lead}>
        This is a fraud detection system built to explore how machine learning and AI can work together to make
        fraud review faster and more transparent, not just automated.
      </p>

      <h2 className={styles.sectionTitle}>What it does</h2>
      <p className={styles.paragraph}>When you submit a transaction, four things happen in sequence:</p>
      <ol className={styles.list}>
        <li>A trained model scores it as fraud or legitimate.</li>
        <li>SHAP, a standard explainability method, breaks down exactly which factors drove that score.</li>
        <li>
          An AI layer retrieves relevant fraud policy guidance and writes a plain-English explanation grounded in
          that evidence.
        </li>
        <li>
          An agent checks the merchant's history and the relevant policy, then recommends whether a human should
          review the case or clear it, logging its reasoning for audit.
        </li>
      </ol>

      <h2 className={styles.sectionTitle}>Built on simulated data</h2>
      <p className={styles.paragraph}>
        This project runs on a public, synthetic dataset of about 1.3 million transactions, generated to resemble
        real card activity. It is not connected to any real bank, payment network, or live transactions. Every
        merchant, cardholder, and city in this demo comes from that same simulated dataset, around 1,000 fictional
        cardholders in total. It cannot evaluate a real transaction and was never intended to.
      </p>

      <p className={styles.note}>
        One practical note: the AI-generated explanation and agent review run on a personal API budget. If you see
        those sections fail to load, it likely means that budget is temporarily exhausted, not that something is
        broken. The core prediction and explainability always work independently of this.
      </p>
    </div>
  );
}
