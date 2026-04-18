import * as FileSystem from 'expo-file-system/legacy';

export async function ensureDir(dirUri: string) {
  const info = await FileSystem.getInfoAsync(dirUri);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(dirUri, { intermediates: true });
  }
}

export async function writeBase64Png(opts: { base64: string; filename: string }): Promise<string> {
  const dir = `${FileSystem.documentDirectory}skinjourney/`;
  await ensureDir(dir);
  const uri = `${dir}${opts.filename}`;
  await FileSystem.writeAsStringAsync(uri, opts.base64, { encoding: FileSystem.EncodingType.Base64 });
  return uri;
}

