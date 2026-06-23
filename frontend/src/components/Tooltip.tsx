import styles from './Tooltip.module.css';

interface Props {
  text: string;
}

/** Small "?" affordance next to a field label; shows an explanation on hover or keyboard focus. */
export function Tooltip({ text }: Props) {
  return (
    <span className={styles.wrap}>
      <button type="button" className={styles.icon} tabIndex={0} aria-label={text}>
        ?
      </button>
      <span className={styles.bubble}>{text}</span>
    </span>
  );
}
