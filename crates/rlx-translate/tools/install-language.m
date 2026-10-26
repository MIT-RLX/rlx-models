// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>

static Class serviceClass(void) {
  static const char *kPath =
      "/System/Library/PrivateFrameworks/TranslationDaemon.framework/TranslationDaemon";
  if (!dlopen(kPath, RTLD_NOW)) {
    fprintf(stderr, "cannot load TranslationDaemon: %s\n", dlerror());
    return Nil;
  }
  Class c = objc_getClass("_LTDLanguageAssetService");
  if (!c) fprintf(stderr, "_LTDLanguageAssetService is missing on this OS\n");
  return c;
}

static void listAssets(Class c) {
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  ((void (*)(id, SEL, id))objc_msgSend)(
      c, sel_getUid("_availableAssetsWithCompletion:"), ^(NSArray *assets) {
        for (id a in assets) printf("  %s\n", [[a description] UTF8String]);
        dispatch_semaphore_signal(sem);
      });
  if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 120LL * NSEC_PER_SEC)))
    fprintf(stderr, "timed out listing assets\n");
}

int main(int argc, char **argv) {
  @autoreleasepool {
    Class c = serviceClass();
    if (!c) return 1;

    if (argc < 2) {
      printf("locales (pass some to install):\n");
      listAssets(c);
      return 0;
    }

    NSMutableArray<NSLocale *> *locales = [NSMutableArray array];
    for (int i = 1; i < argc; i++)
      [locales addObject:[NSLocale localeWithLocaleIdentifier:@(argv[i])]];
    printf("installing %s ...\n", [[locales description] UTF8String]);

    __block NSString *last = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    ((void (*)(id, SEL, id, BOOL, id, id))objc_msgSend)(
        c, sel_getUid("setInstalledLocales:useCellular:progress:completion:"), locales, YES,
        ^(NSArray *progress) {
          // Progress fires often; only print when the summary line changes.
          NSMutableString *line = [NSMutableString string];
          for (id p in progress) {
            NSString *d = [p description];
            NSRange nl = [d rangeOfString:@"\n"];
            [line appendFormat:@"%@ ", nl.location == NSNotFound
                                           ? d
                                           : [d substringToIndex:nl.location]];
          }
          if (![line isEqualToString:last]) {
            printf("  %s\n", line.UTF8String);
            fflush(stdout);
            last = line;
          }
        },
        ^(id result, NSError *err) {
          if (err) fprintf(stderr, "error: %s\n", [[err description] UTF8String]);
          dispatch_semaphore_signal(sem);
        });

    // A cold locale is several hundred megabytes.
    if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 3600LL * NSEC_PER_SEC))) {
      fprintf(stderr, "timed out after an hour\n");
      return 1;
    }
    printf("\nfinal state:\n");
    listAssets(c);
  }
  return 0;
}
